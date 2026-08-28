"""FastAPI app assembly. The OpenAI-compatible shim and the MCP adapter are
both mounted here; REST is still a real, deliberately deferred adapter
(README "Architecture"), not a missing piece of this file.

Owns two things beyond routing: the connection queries are served from, and a
background loop that indexes on start and then every `index.refresh_interval_s`
(README "Deployment"). The index loop uses its **own** connection, separate
from the one serving queries — `vault_ask.db.connect` turns on WAL mode for
exactly this: a writer mid-transaction must not block, or be interleaved with,
a concurrent reader on another connection.

MCP's streamable-HTTP app carries its own ASGI lifespan (it starts a session
manager), which `Mount()` alone does not run — FastAPI only drives its own
lifespan for a mounted sub-app, not the sub-app's. Combined explicitly below
with `AsyncExitStack`, the documented pattern for embedding an MCP server
inside an existing app rather than running it standalone.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, RedirectResponse

from .. import __version__
from ..config import Settings, overrides_path
from ..db import connect
from ..ingest import run_ingest
from ..overrides import read_overrides
from ..vault import LiveSyncVault
from . import mcp_adapter
from .admin import router as admin_router
from .openai_shim import router as openai_router

STATIC = Path(__file__).resolve().parent / "static"

log = logging.getLogger("vault_ask.api")


def _default_gateway() -> str | None:
    """This container's IPv4 default gateway, from /proc/net/route.

    Parsed rather than assumed. The macvlan this deployment uses is a /25, so
    its gateway is not the `.1` a home network's usual convention suggests —
    and aiming the wake below at a guessed address is a silent no-op that looks
    exactly like the fix not working. Ask the routing table instead.
    """
    try:
        with open("/proc/net/route") as fh:
            for line in fh.readlines()[1:]:
                fields = line.split()
                # destination 00000000 == default route; gateway is little-endian hex
                if len(fields) > 2 and fields[1] == "00000000":
                    raw = int(fields[2], 16)
                    return ".".join(str((raw >> (8 * i)) & 0xFF) for i in range(4))
    except OSError:
        return None
    return None


async def _wake_arp() -> None:
    """Send one outbound packet so the LAN can reach us after a restart.

    Containers on this NAS's macvlan come back from a recreate *healthy and
    unreachable*: the healthcheck runs against localhost inside the container
    and passes, while the router keeps answering for the pre-restart ARP entry
    for minutes. One outbound packet from inside the container re-triggers
    resolution and it recovers instantly (see README "Deployment").

    Done here because restarting is now the documented way to apply an admin
    console change — and someone editing config in a browser cannot be expected
    to shell into the NAS and run `docker exec` afterwards. Entirely
    best-effort: the connection is *expected* to fail (nothing need listen on
    the gateway); sending the SYN is the whole point.
    """
    gateway = _default_gateway()
    if gateway is None:
        log.debug("arp_wake.skipped reason=no_default_route")
        return
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(gateway, 80), timeout=3.0
        )
        writer.close()
    except (TimeoutError, OSError):
        pass  # refused/unreachable is fine — the packet left, which is all we need
    log.info("arp_wake.sent gateway=%s", gateway)


async def _index_once(cfg: Settings) -> None:
    vault = LiveSyncVault(
        cfg.vault,
        cfg.vault_couchdb_password.get_secret_value() if cfg.vault_couchdb_password else None,
    )
    try:
        conn = connect(cfg.index.db_path, embedding_dim=cfg.models.embedding_dim)
        try:
            report = await run_ingest(vault, conn, cfg)
            log.info("index_loop.done\n%s", report.summary())
        finally:
            conn.close()
    except Exception:
        # A scheduled job must outlive its runs — same stance as
        # clippings-topics' serve(): log it, try again next tick.
        log.exception("index_loop.failed")
    finally:
        await vault.close()


async def _index_loop(cfg: Settings) -> None:
    if not cfg.vault.couchdb_url:
        log.warning("index_loop.disabled reason=vault_couchdb_url_unset")
        return

    run_first = cfg.index.run_on_startup
    while True:
        if run_first:
            await _index_once(cfg)
        run_first = True
        await asyncio.sleep(cfg.index.refresh_interval_s)


def create_app(cfg: Settings) -> FastAPI:
    # streamable_http_path="/", mounted below at "/mcp": mounting the sub-app
    # AT "/" instead (with the path baked in as "/mcp") looked equivalent but
    # is not — Mount("/") is a catch-all that intercepts every path on the
    # outer app before /healthz or /v1/* ever get a chance (found by the test
    # suite: /healthz started 404ing the moment this app existed).
    #
    # host="0.0.0.0" is load-bearing, not cosmetic: the SDK auto-enables DNS-
    # rebinding protection (an allowed-Host allowlist of only localhost
    # variants) whenever this looks like a loopback-bound server, which would
    # 421 every real LAN request the moment this is bound to cfg.api.host.
    mcp_app = mcp_adapter.server.streamable_http_app(
        streamable_http_path="/",
        host="0.0.0.0",  # noqa: S104 — bound to the container's LAN IP by compose, like cfg.api.host
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.cfg = cfg
        app.state.conn = connect(cfg.index.db_path, embedding_dim=cfg.models.embedding_dim)
        mcp_adapter.configure(app.state.conn, cfg)
        app.state.admin_api_key = (
            cfg.admin_api_key.get_secret_value() if cfg.admin_api_key else None
        )
        # Snapshot of overrides.json as it stood when this process booted —
        # `pending_restart` in the admin API compares the file's current
        # content against this, not against the shipped defaults, so a save
        # that exactly restates what's already running does not claim a
        # restart is needed.
        app.state.overrides_at_boot = read_overrides(overrides_path())
        # Before the index loop: indexing can take minutes on a cold cache,
        # and the whole point is to be reachable from the LAN promptly after a
        # restart, not once the first ingest finishes.
        await _wake_arp()
        index_task = asyncio.create_task(_index_loop(cfg))
        log.info(
            "api.started db_path=%s host=%s port=%d", cfg.index.db_path, cfg.api.host, cfg.api.port
        )
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(mcp_app.router.lifespan_context(mcp_app))
            try:
                yield
            finally:
                index_task.cancel()
                with suppress(asyncio.CancelledError):
                    await index_task
                app.state.conn.close()

    app = FastAPI(title="vault-ask", version=__version__, lifespan=lifespan)
    app.include_router(openai_router)
    app.include_router(admin_router)
    app.mount("/mcp", mcp_app)

    @app.get("/admin", summary="Admin console (static page; data behind it needs the API key)")
    async def admin_page() -> FileResponse:
        return FileResponse(STATIC / "admin.html")

    @app.get("/chat", summary="Chat UI — the reason the OpenAI shim exists, served in-process")
    async def chat_page() -> FileResponse:
        return FileResponse(STATIC / "chat.html")

    @app.get("/", summary="Land on the chat UI rather than a bare 404")
    async def root() -> RedirectResponse:
        return RedirectResponse("/chat")

    @app.get("/healthz", summary="Liveness check")
    async def healthz() -> dict[str, str]:
        try:
            app.state.conn.execute("SELECT 1")
        except Exception as exc:
            log.warning("healthz.index_probe_failed error=%s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="index unreachable"
            ) from exc
        return {"status": "ok"}

    return app
