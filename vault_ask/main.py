"""vault-ask CLI.

    python -m vault_ask index               incremental index build
    python -m vault_ask index --rebuild     full re-embed (see ingest.run_ingest)
    python -m vault_ask index --dry-run     report what would change, write nothing
    python -m vault_ask ask "question"      keyword retrieval + generation
    python -m vault_ask ask "..." --dry-run retrieval only, no generation
    python -m vault_ask ask "..." --dry-run --json   ranked hits as JSON, with
                                            each hit's source — for A/B runs
                                            (e.g. retrieval.graph_enabled on/off)
    python -m vault_ask serve               FastAPI: OpenAI shim + MCP + background indexing

The REST adapter lands later (README "Architecture") — ``serve`` only mounts
what's actually implemented.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import uvicorn

from .api.app import create_app
from .ask import ask as run_ask
from .ask import hits_as_json
from .config import load_settings
from .db import connect
from .embed import EmbeddingSpaceChanged
from .ingest import MassDeletionRefused, run_ingest
from .vault import LiveSyncVault, VaultUnavailable

log = logging.getLogger("vault-ask")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="vault-ask")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="build or refresh the index from the vault")
    index.add_argument(
        "--rebuild", action="store_true", help="ignore the cache, re-read every note"
    )
    index.add_argument(
        "--dry-run", action="store_true", help="report what would change; write nothing"
    )
    index.add_argument(
        "--allow-mass-delete",
        action="store_true",
        help="permit a run that deletes a large share of the index "
        "(see ingest.MassDeletionRefused)",
    )

    ask = sub.add_parser("ask", help="answer a question from the index")
    ask.add_argument("question")
    ask.add_argument(
        "--dry-run", action="store_true", help="retrieval only, no generation"
    )
    ask.add_argument(
        "--json",
        action="store_true",
        help="with --dry-run: structured retrieval results — full score precision "
        "and each hit's source (vector/fts/graph), for A/B comparison",
    )
    ask.add_argument(
        "--no-web",
        dest="allow_web",
        action="store_false",
        default=True,
        help="see the whole vault, including `personal` chunks (default: `open` only)",
    )

    sub.add_parser("serve", help="run the OpenAI-compatible API, indexing in the background")

    return parser.parse_args()


async def _index(args: argparse.Namespace) -> int:
    cfg = load_settings(args.config)
    if not cfg.vault.couchdb_url:
        log.error("vault.couchdb_url is not set (VAULTASK_VAULT__COUCHDB_URL)")
        return 2

    vault = LiveSyncVault(
        cfg.vault,
        cfg.vault_couchdb_password.get_secret_value() if cfg.vault_couchdb_password else None,
    )
    try:
        conn = connect(cfg.index.db_path, embedding_dim=cfg.models.embedding_dim)
        try:
            report = await run_ingest(
                vault,
                conn,
                cfg,
                dry_run=args.dry_run,
                rebuild=args.rebuild,
                allow_mass_delete=args.allow_mass_delete,
            )
        finally:
            conn.close()
    except VaultUnavailable as exc:
        log.error("vault unavailable: %s", exc)
        return 1
    except EmbeddingSpaceChanged as exc:
        log.error("%s", exc)
        return 3
    except MassDeletionRefused as exc:
        log.error("%s", exc)
        return 4
    finally:
        await vault.close()

    mode = "dry-run — nothing written" if args.dry_run else "applied"
    print(f"index ({mode}):")
    print(report.summary())
    return 0


async def _ask(args: argparse.Namespace) -> int:
    cfg = load_settings(args.config)
    conn = connect(cfg.index.db_path, embedding_dim=cfg.models.embedding_dim)
    try:
        answer = await run_ask(
            conn, cfg, args.question, allow_web=args.allow_web, dry_run=args.dry_run
        )
    finally:
        conn.close()

    if args.dry_run and args.json:
        print(hits_as_json(answer.hits))
    else:
        print(answer.text)
    return 0


def _serve(args: argparse.Namespace) -> None:
    cfg = load_settings(args.config)
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port, log_config=None)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    args = _args()
    match args.command:
        case "index":
            sys.exit(asyncio.run(_index(args)))
        case "ask":
            sys.exit(asyncio.run(_ask(args)))
        case "serve":
            _serve(args)
        case _:  # pragma: no cover - argparse already restricts this
            raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    main()
