"""OpenAI-compatible shim: `POST /v1/chat/completions`, `GET /v1/models`.

The priority adapter (README "Architecture") — the only one of the three that
comes with a UI for free: point Open WebUI or LibreChat at this and there is a
working chat interface with no frontend of this project's own to write.

Advertises one model id, `vault-ask`. Streaming is required (Open WebUI is
unpleasant without it) and implemented as real token streaming from the
generation model (`vault_ask.ask.ask_stream`), not a pre-computed answer typed
out artificially — the point of streaming is time-to-first-token, which a
fake stream over an already-finished answer would not provide.

``allow_web`` defaults to **false** here, unlike the REST adapter's planned
default of true (README "Corpus and sensitivity") — a model on the other end
of this endpoint is not one this application controls, the same reasoning
that applies to MCP.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..ask import ask, ask_stream
from ..config import Settings

log = logging.getLogger("vault_ask.api.openai_shim")

router = APIRouter()

MODEL_ID = "vault-ask"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    #: Not part of the OpenAI schema — an escape hatch for tests and power
    #: users. Real clients (Open WebUI) never send this, so it defaults false.
    allow_web: bool = False


def _last_user_message(messages: list[ChatMessage]) -> str | None:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return None


@router.get("/v1/models")
async def list_models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "vault-ask"}],
    }


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: Request, body: ChatCompletionRequest
) -> JSONResponse | StreamingResponse:
    conn = request.app.state.conn
    cfg: Settings = request.app.state.cfg

    question = _last_user_message(body.messages)
    if question is None:
        raise HTTPException(status_code=400, detail="`messages` has no message with role 'user'")

    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    if body.stream:
        return StreamingResponse(
            _stream(conn, cfg, question, allow_web=body.allow_web, request_id=request_id),
            media_type="text/event-stream",
        )

    answer = await ask(conn, cfg, question, allow_web=body.allow_web)
    return JSONResponse(
        {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer.text},
                    "finish_reason": "stop",
                }
            ],
            # Not tracked — vault-ask does not meter itself the way a paid API
            # does. Present because some clients assume the key exists.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


async def _stream(
    conn: sqlite3.Connection, cfg: Settings, question: str, *, allow_web: bool, request_id: str
) -> AsyncIterator[str]:
    created = int(time.time())

    def sse(delta: dict[str, str], finish_reason: str | None = None) -> str:
        payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield sse({"role": "assistant"})
    async for delta in ask_stream(conn, cfg, question, allow_web=allow_web):
        yield sse({"content": delta})
    yield sse({}, finish_reason="stop")
    yield "data: [DONE]\n\n"
