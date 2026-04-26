# ChatMCP Voice MCP

End-to-end speech-to-speech voice plugin for [ChatMCP](https://github.com/AvengerMoJo/chatmcp) using MiniCPM-o 2.6.

```
Audio in → MiniCPM-o 2.6 (Whisper + Qwen3 + CosyVoice2) → Audio out
                         ↓
                    [bridge] trigger → ChatMCP backend routing
```

## Quick Start

```bash
git clone https://github.com/AvengerMoJo/chatmcp-voice-mcp.git
cd chatmcp-voice-mcp

# Install dependencies
pip install faster-whisper edge-tts numpy torch transformers soundfile

# Optional: install ffmpeg for WebM conversion
apt install ffmpeg  # Linux
brew install ffmpeg # macOS

# Start server (bridge-only mode, no GPU needed)
python3 voice_http_server.py --no-model --port 9090 --https
```

Open `https://localhost:9090/` in your browser to use the voice recorder.

## Two Modes

### Mode 1: MiniCPM-o 2.6 (end-to-end)

Single model handles everything — Whisper encode, Qwen3 think, CosyVoice2 decode — fused in one forward pass. Requires ~15GB VRAM.

```bash
python3 voice_http_server.py --model openbmb/MiniCPM-o-2_6 --port 9090 --https
```

During Qwen3's text-thinking phase, `[bridge]...[/bridge]` triggers route commands to ChatMCP's backend LLM. When the result returns, it's injected back into context and MiniCPM-o continues speaking.

### Mode 2: Cascaded fallback (`--no-model`)

Separate components that work without GPU:

| Stage | Model | Install |
|-------|-------|---------|
| ASR | faster-whisper tiny | `pip install faster-whisper` |
| LLM | Ollama (any model) | `ollama pull llama3.2:3b` |
| TTS | edge-tts | `pip install edge-tts` |

```bash
python3 voice_http_server.py --no-model --port 9090 --https
```

Configure via environment variables:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.2:3b
export EDGE_TTS_VOICE=en-US-JennyNeural
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI (voice recorder) |
| `GET` | `/health` | Server status |
| `GET` | `/tools` | Available tools |
| `POST` | `/chat` | Text in, text out |
| `POST` | `/voice_upload` | Audio file in, audio file out |
| `POST` | `/bridge` | Inject backend result |

## API Examples

```bash
# Health check
curl https://localhost:9090/health

# Text chat
curl -X POST https://localhost:9090/chat \
  -H "Content-Type: application/json" \
  -d '{"text":"What is the weather?"}'

# Voice upload (WAV file in, WAV file out)
curl -X POST https://localhost:9090/voice_upload \
  -F "audio=@speech.wav" \
  -o response.wav

# Bridge result injection
curl -X POST https://localhost:9090/bridge \
  -H "Content-Type: application/json" \
  -d '{"result":"The deadline is next Friday"}'
```

## MCP Integration

When installed as a ChatMCP plugin, this server speaks JSON-RPC 2.0 over stdio and exposes three tools:

- `voice_query_text` — Text through MiniCPM-o with [bridge] detection
- `voice_query_audio` — Speech WAV through MiniCPM-o end-to-end (audio in, audio out)
- `voice_bridge_result` — Inject backend result back into voice context

ChatMCP connects to it via the MCP protocol just like any other MCP server.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Local Machine                                       │
│                                                      │
│  ┌──────────────────────┐       ┌─────────────────┐ │
│  │    MiniCPM-o 2.6     │       │    ChatMCP       │ │
│  │    (voice_mcp.py)   │       │    (MCP client)  │ │
│  │                      │       │                  │ │
│  │  Whisper encode ←────┼───────┤                  │ │
│  │       ↓              │       │                  │ │
│  │  Qwen3 text think ───┤ [bridge]→ MCP routing   │ │
│  │       ↓              │       │       ↓          │ │
│  │  CosyVoice2 decode ──┤←──────┤  Backend LLM     │ │
│  │       ↓              │       │                  │ │
│  │  Audio out ──────────┤       │                  │ │
│  └──────────────────────┘       └─────────────────┘ │
└─────────────────────────────────────────────────────┘
```

## File Structure

```
chatmcp-voice-mcp/
├── models/
│   ├── minicpm_o_model.py    # MiniCPM-o 2.6 inference (real, end-to-end)
│   └── minicpm_wrapper.py    # (deprecated) stub wrapper
├── voice_mcp_server.py       # MCP server (JSON-RPC over stdio)
├── voice_http_server.py      # HTTP server (browser + curl testing)
├── voice_pipeline.py         # Fallback: ASR→LLM→TTS (--no-model mode)
├── voice_test.html           # Browser voice recorder UI
├── test_voice_server.sh      # Curl test script
├── setup.sh                  # Dependencies installer
├── plugin.yaml               # Plugin manifest
├── requirements.txt
└── LICENSE
```

## Troubleshooting

**Mic doesn't work in browser:** Make sure you open `https://localhost:9090/` (not `file://` and not `http://` on remote IP). Browsers require secure context for `getUserMedia`.

**Self-signed cert warning:** The `--https` flag auto-generates a cert. Click "Advanced → Proceed anyway" in your browser.

**WebM uploads fail:** Install ffmpeg for automatic format conversion. Without it, only WAV files are accepted.

**Model won't load:** MiniCPM-o 2.6 requires ~15GB VRAM. Use `--no-model` for CPU-only fallback mode.
