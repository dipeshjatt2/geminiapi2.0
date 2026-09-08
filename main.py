"""OpenAI-compatible HTTP bridge in front of cookie-based Gemini.

Run with:  uvicorn main:app --host 0.0.0.0 --port 8787

Endpoints
---------
- GET  /healthz                       Liveness / readiness probe
- GET  /v1/models                     Model list (OpenAI-compatible)
- POST /v1/chat/completions           Chat completions, streaming + non-streaming

Point Kimi Code (or any OpenAI-compatible client) at ``http://<host>:8787/v1``
with the ``BRIDGE_API_KEY`` as the API key.
"""

import asyncio
import json
import logging
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

import gemini_backend

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("gemini-bridge-http")

app = FastAPI(title="gemini-webapi openai bridge", version="1.0.0")

API_KEY = (os.environ.get("BRIDGE_API_KEY") or "").strip()
MODEL_ID = (os.environ.get("BRIDGE_MODEL_ID") or "gemini-web").strip()
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "900"))
_MAX_TRANSCRIPT_CHARS = int(os.environ.get("MAX_TRANSCRIPT_CHARS", "1500000"))


def _authorized(request: Request) -> bool:
    if not API_KEY:
        return True
    header = request.headers.get("authorization") or ""
    return header == f"Bearer {API_KEY}"


def _error(status: int, message: str, code: str = "invalid_request_error"):
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": code,
                "param": None,
                "code": code,
            }
        },
    )


def _cut_transcript(messages: list) -> list:
    """Keep the transcript under a char budget (drop oldest user/assistant)."""
    total = 0
    kept = []
    for message in reversed(messages):
        text = gemini_backend._content_text(message.get("content"))
        cost = len(text) + 80
        if kept and total + cost > _MAX_TRANSCRIPT_CHARS:
            continue
        kept.append(message)
        total += cost
    return list(reversed(kept))


# --------------------------------------------------------------------------- #
# Chat completion plumbing
# --------------------------------------------------------------------------- #

def _now() -> int:
    return int(time.time())


def _message_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:16]}"


async def _run_backend(messages: list, tools: list):
    """Call the cookie backend inside the request timeout."""
    return await asyncio.wait_for(
        gemini_backend.complete(messages, tools), timeout=REQUEST_TIMEOUT
    )


def _non_stream_response(request_id: str, model: str, result: dict) -> dict:
    message = {"role": "assistant", "content": None}
    finish_reason = "stop"
    if result["kind"] == "tool_calls":
        message["tool_calls"] = result["tool_calls"]
        finish_reason = "tool_calls"
    else:
        message["content"] = result["text"]
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
    }


def _sse_payload(request_id: str, model: str, delta: dict, finish_reason=None):
    choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    return (
        "data: "
        + json.dumps(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": _now(),
                "model": model,
                "choices": [choice],
            },
            ensure_ascii=False,
        )
        + "\n\n"
    )


def _chunk_text(text: str, size: int = 120):
    for index in range(0, len(text), size):
        yield text[index : index + size]


async def _stream_result(request_id: str, model: str, result: dict):
    yield _sse_payload(
        request_id,
        model,
        {"role": "assistant", "content": ""},
    )
    if result["kind"] == "tool_calls":
        calls = result["tool_calls"]
        for index, call in enumerate(calls):
            function = call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            yield _sse_payload(
                request_id,
                model,
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call.get("id"),
                            "type": "function",
                            "function": {
                                "name": function.get("name"),
                                "arguments": "",
                            },
                        }
                    ]
                },
            )
            for piece in _chunk_text(arguments):
                yield _sse_payload(
                    request_id,
                    model,
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "function": {"arguments": piece},
                            }
                        ]
                    },
                )
        yield _sse_payload(request_id, model, {}, finish_reason="tool_calls")
    else:
        for piece in _chunk_text(result["text"]):
            yield _sse_payload(request_id, model, {"content": piece})
        yield _sse_payload(request_id, model, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "cookies_configured": gemini_backend.cookies_configured(),
        "model_id": MODEL_ID,
    }


@app.get("/v1/models")
async def list_models(request: Request):
    if not _authorized(request):
        return _error(401, "Invalid bearer token", "invalid_api_key")
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "gemini-webapi",
                "created": 0,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not _authorized(request):
        return _error(401, "Invalid bearer token", "invalid_api_key")

    try:
        body = await request.json()
    except Exception:
        return _error(400, "Request body is not valid JSON")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return _error(400, "messages must be a non-empty array")

    stream = bool(body.get("stream", False))
    model = str(body.get("model") or MODEL_ID)
    tools = body.get("tools") or []
    if not isinstance(tools, list):
        tools = []

    messages = _cut_transcript(messages)

    request_id = _message_id()
    try:
        result = await _run_backend(messages, tools)
    except asyncio.TimeoutError:
        return _error(504, "Gemini request timed out", "request_timeout")
    except RuntimeError as exc:
        log.warning("Backend failure: %s", exc)
        return _error(502, str(exc), "upstream_error")
    except Exception as exc:
        log.exception("Unexpected backend failure")
        return _error(502, f"Gemini request failed: {exc}", "upstream_error")

    if stream:
        return StreamingResponse(
            _stream_result(request_id, model, result),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    return JSONResponse(_non_stream_response(request_id, model, result))
