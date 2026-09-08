# gemini-api-2.0 bridge

Exposes your **cookie-authenticated Gemini** (via
[`gemini_webapi`](https://github.com/HanaokaYuzu/Gemini-API)) as an
**OpenAI-compatible HTTP API**, so coding agents that speak the OpenAI Chat
Completions protocol (Kimi Code, etc.) can use it as a model backend.

The bridge is stateless: every request re-sends the full conversation to Gemini
as one prompt (labelled `USER:` / `ASSISTANT:` / `TOOL RESULT:` transcript).
The Gemini account's default model is used unless `GEMINI_MODEL` is set.

```
user (Kimi Code / agent)
   │  OpenAI chat.completions (JSON / SSE)
   ▼
main.py  ──>  gemini_backend.py  ──>  gemini_webapi (cookie auth)  ──>  gemini.google.com
```

## What is supported

- `POST /v1/chat/completions` — streaming (SSE) and non-streaming.
- `GET /v1/models`, `GET /healthz`.
- Text conversations with full history; images (last user message, `image_url`
  as `data:` URL or public http(s) URL) are attached to Gemini.
- **Tool calls are emulated.** The web app cannot natively do OpenAI function
  calling, so when a request declares `tools`, a prompt contract asks Gemini to
  reply with a JSON `{"tool_calls": [...]}` object when it wants to run a tool.
  The bridge converts that into the OpenAI `tool_calls` response shape, so
  agents can still read/edit files. It is prompt-level and can be fooled by
  prompt injection — use it only with content you trust.

## Known limitations

- Cookie accounts are shared: all bridge users consume the same Google account.
- The web app has its own rate limits and cookie lifetimes; `__Secure-1PSIDTS`
  is auto-refreshed and persisted under `DATA_DIR`.
- Tool-call emulation depends on Gemini following the JSON contract; occasional
  malformed tool replies are possible (the agent will usually retry).
- No `usage` counters, no `temperature`/`max_tokens` mapping.
- Gemini web responses cannot be truly token-streamed: text is generated fully
  and then split into SSE chunks.

## Run

```bash
cd geminiapi2.0

# cookies: export __Secure-1PSID / __Secure-1PSIDTS from gemini.google.com
export GEMINI_PSID='...'
export GEMINI_PSIDTS='...'          # optional but recommended
export BRIDGE_API_KEY='change-me'

docker build -t gemini-bridge .
docker run -p 8787:8787 \
  -e GEMINI_PSID="$GEMINI_PSID" \
  -e GEMINI_PSIDTS="$GEMINI_PSIDTS" \
  -e BRIDGE_API_KEY="$BRIDGE_API_KEY" \
  -v gemini-bridge-data:/data \
  gemini-bridge
```

Or run without Docker:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8787
```

Smoke test:

```bash
curl http://127.0.0.1:8787/healthz
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-web","messages":[{"role":"user","content":"hi, reply in one line"}]}'
```

## Configure Kimi Code (`config.toml`)

Add to `~/.kimi-code/config.toml` (no registry URL needed):

```toml
default_model = "gemini-web/gemini-web"

[providers."gemini-web"]
type = "openai"
base_url = "http://127.0.0.1:8787/v1"
api_key = "change-me"

[models."gemini-web/gemini-web"]
provider = "gemini-web"
model = "gemini-web"
max_context_size = 1000000
capabilities = ["image_in", "tool_use"]
display_name = "Gemini Web (cookie)"
```

If the bridge runs on another machine, replace `127.0.0.1` with its address and
make sure the port is reachable (the bridge does not add TLS — put it behind a
reverse proxy if you need https).

Run: `kimi -m gemini-web/gemini-web`

Notes:

- `max_context_size` can be lowered if you prefer earlier compaction.
- `model` must match `BRIDGE_MODEL_ID` (default `gemini-web`); the value is only
  echoed back, the wrapper uses the account default model unless
  `GEMINI_MODEL` is set.
- Leave `BRIDGE_API_KEY` empty only if the port is bound to localhost and no
  other user can reach it.
