"""Cookie-based Gemini backend for the OpenAI-compatible bridge.

Thin wrapper around ``gemini_webapi`` (HanaokaYuzu/Gemini-API).  Authentication
uses shared Google cookies (``__Secure-1PSID`` / ``__Secure-1PSIDTS``) provided
through environment variables or a JSON cookie file, mirroring how the
FreePyQuizBot uses the same library.

Because the Gemini web app does not expose OpenAI-style function calling, tool
calls are emulated at the prompt level: when a request declares ``tools``, a
hidden contract asks the model to answer with a single JSON object shaped like
``{"tool_calls": [...]}`` when it wants to run a tool.  The response is parsed
back into the OpenAI ``tool_calls`` structure here.
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import tempfile

log = logging.getLogger("gemini-bridge")

try:
    from gemini_webapi import GeminiClient
except Exception as exc:  # pragma: no cover - deployment specific
    GeminiClient = None
    log.warning("gemini_webapi is not installed: %s", exc)

# Deterministic extension map (containers may lack /etc/mime.types).
_FILE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "application/pdf": ".pdf",
}

_client = None
_lock = asyncio.Lock()


def cookies_configured() -> bool:
    """Return True when usable cookies exist in env or the cookie file."""
    psid, _psidts = _load_cookies()
    return bool(psid)


def _load_cookies():
    psid = (os.environ.get("GEMINI_PSID") or "").strip()
    psidts = (os.environ.get("GEMINI_PSIDTS") or "").strip()
    if psid:
        return psid, psidts
    cookie_file = os.environ.get("GEMINI_COOKIE_FILE")
    if cookie_file and os.path.exists(cookie_file):
        try:
            with open(cookie_file, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            return (
                str(data.get("__Secure-1PSID") or "").strip(),
                str(data.get("__Secure-1PSIDTS") or "").strip(),
            )
        except Exception as exc:
            log.warning("Could not read cookie file %s: %s", cookie_file, exc)
    return "", ""


async def ensure_client():
    """Lazily build and initialise the shared GeminiClient (cached)."""
    global _client
    if GeminiClient is None:
        raise RuntimeError("gemini_webapi is not installed in this container")
    if _client is not None:
        return _client
    async with _lock:
        if _client is None:
            psid, psidts = _load_cookies()
            if not psid:
                raise RuntimeError(
                    "No Gemini cookies configured. Set GEMINI_PSID / GEMINI_PSIDTS "
                    "(or GEMINI_COOKIE_FILE) first."
                )
            cookie_path = os.environ.get("DATA_DIR") or "/data"
            os.environ.setdefault("GEMINI_COOKIE_PATH", cookie_path)
            os.makedirs(os.environ["GEMINI_COOKIE_PATH"], exist_ok=True)
            client = GeminiClient(secure_1psid=psid, secure_1psidts=psidts, proxy=None)
            await client.init(
                timeout=600, auto_close=False, close_delay=300, auto_refresh=True
            )
            _client = client
        return _client


async def generate_text(prompt: str, files=None, model: str = None) -> str:
    """Run one prompt through the cookie Gemini web API and return its text."""
    async with _lock:
        client = await ensure_client()
        output = await client.generate_content(
            prompt, files=files or None, model=model or None, temporary=True
        )
    text = (output.text or "").strip() if output is not None else ""
    if not text:
        raise RuntimeError("Gemini returned an empty response (cookies may be expired)")
    return text


# --------------------------------------------------------------------------- #
# Message / tool-call conversion
# --------------------------------------------------------------------------- #

def _content_text(content) -> str:
    """Extract plain text from a string or an OpenAI content part list."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("text", "input_text"):
                value = part.get("text")
                if isinstance(value, str):
                    pieces.append(value)
        return "\n".join(pieces)
    return str(content)


def _flatten_messages(messages: list) -> str:
    """Render the OpenAI message history as a labelled transcript."""
    lines = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = _content_text(content)
            if text.strip():
                lines.append(f"SYSTEM: {text.strip()}")
        elif role == "user":
            text = _content_text(content)
            if text.strip():
                lines.append(f"USER: {text.strip()}")
        elif role == "assistant":
            text = _content_text(content)
            tool_calls = message.get("tool_calls") or []
            if text.strip():
                lines.append(f"ASSISTANT: {text.strip()}")
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                name = function.get("name") or "tool"
                arguments = function.get("arguments") or "{}"
                lines.append(f"ASSISTANT TOOL CALL: {name}({arguments})")
        elif role == "tool":
            name = message.get("name") or "tool"
            text = _content_text(content)
            lines.append(f"TOOL RESULT ({name}): {text.strip() or '(no output)'}")
    return "\n".join(line for line in lines if line)


def _tool_contract(tools: list) -> str:
    """Hidden prompt contract that lets the model request tool calls."""
    descriptions = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") or {}
        name = function.get("name") or "unknown"
        description = (function.get("description") or "").strip()
        parameters = function.get("parameters") or {}
        schema = json.dumps(parameters, ensure_ascii=False) if parameters else "{}"
        descriptions.append(
            f"- {name}: {description} (arguments JSON schema: {schema})"
        )
    listing = "\n".join(descriptions) if descriptions else "(none declared)"
    return (
        "AVAILABLE TOOLS\n"
        "You may use the following tools to complete the user's request:\n"
        f"{listing}\n\n"
        "If a tool is required, answer with a SINGLE raw JSON object and nothing "
        "else (no markdown, no code fences, no prose). It must have exactly this shape:\n"
        '{"tool_calls": [{"id": "call_1", "type": "function", "function": '
        '{"name": "TOOL_NAME", "arguments": {ARGS AS A JSON OBJECT}}}]}\n'
        'If several tools are needed at once, list them all inside "tool_calls". '
        "Otherwise answer normally with plain text."
    )


def _extract_json_object(text: str):
    """Return the first balanced top-level JSON object in text, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _parse_response(raw_text: str) -> dict:
    """Decide whether the model asked for tool calls or answered in text."""
    candidate = _extract_json_object(raw_text)
    if isinstance(candidate, dict):
        calls = candidate.get("tool_calls")
        if isinstance(calls, list) and calls:
            parsed_calls = []
            for number, call in enumerate(calls, start=1):
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                name = function.get("name")
                if not name:
                    continue
                arguments = function.get("arguments")
                if isinstance(arguments, (dict, list)):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif arguments is None:
                    arguments = "{}"
                else:
                    arguments = str(arguments)
                parsed_calls.append(
                    {
                        "id": call.get("id") or f"call_{number}",
                        "type": "function",
                        "function": {"name": str(name), "arguments": arguments},
                    }
                )
            if parsed_calls:
                return {"kind": "tool_calls", "tool_calls": parsed_calls}
    return {"kind": "text", "text": raw_text}


async def _fetch_remote_image(url: str):
    """Download an http(s) image, returning (bytes, mime) or None."""
    try:
        from curl_cffi.requests import AsyncSession

        async with AsyncSession(timeout=30) as session:
            response = await session.get(url)
            if response.status_code != 200:
                return None
            mime_type = (
                response.headers.get("content-type") or "image/png"
            ).split(";")[0].strip().lower()
            return response.content, mime_type
    except Exception as exc:
        log.warning("Could not download image %s: %s", url[:80], exc)
        return None


async def _first_image(content) -> "tuple[bytes, str] | None":
    """Return (bytes, mime) for the first image part of a content list."""
    if not isinstance(content, list):
        return None
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in ("image_url", "image"):
            continue
        image_url = part.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if not isinstance(url, str) or not url:
            continue
        if url.startswith("data:"):
            try:
                header, _, payload = url.partition(",")
                mime_type = (
                    header[5:].split(";")[0].strip().lower() or "image/png"
                )
                return base64.b64decode(payload), mime_type
            except Exception as exc:
                log.warning("Could not decode data URL image: %s", exc)
                continue
        if url.startswith(("http://", "https://")):
            return await _fetch_remote_image(url)
    return None


async def complete(messages: list, tools: list) -> dict:
    """Run the OpenAI-style conversation through the cookie Gemini backend.

    Returns ``{"kind": "text", "text": ...}`` or
    ``{"kind": "tool_calls", "tool_calls": [...]}``.
    """
    system_parts = [
        _content_text(message.get("content")).strip()
        for message in messages
        if message.get("role") == "system"
    ]
    system_block = (
        "\n".join(part for part in system_parts if part)
        if any(system_parts)
        else "You are a helpful, accurate coding assistant."
    )

    history = _flatten_messages(
        [message for message in messages if message.get("role") != "system"]
    )
    transcript = f"SYSTEM INSTRUCTIONS:\n{system_block.strip()}"
    if history:
        transcript += f"\n\n--- CONVERSATION ---\n{history}"
    if tools:
        transcript += f"\n\n{_tool_contract(tools)}"
    transcript += "\n\nRespond now."

    model = os.environ.get("GEMINI_MODEL") or None

    files = None
    user_messages = [message for message in messages if message.get("role") == "user"]
    image = await _first_image(user_messages[-1].get("content")) if user_messages else None

    temp_paths = []
    try:
        if image is not None:
            image_bytes, mime_type = image
            extension = (
                _FILE_EXTENSIONS.get(mime_type)
                or mimetypes.guess_extension(mime_type)
                or ".png"
            )
            handle = tempfile.NamedTemporaryFile(
                dir=os.environ.get("TMP_DIR") or "/tmp",
                prefix="bridge_img_",
                suffix=extension,
                delete=False,
            )
            handle.write(image_bytes)
            handle.close()
            temp_paths.append(handle.name)
            files = [handle.name]
        raw_text = await generate_text(transcript, files=files, model=model)
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass
    return _parse_response(raw_text)
