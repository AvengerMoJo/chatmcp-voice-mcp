#!/usr/bin/env bash
# Install dependencies for real ASR + LLM + TTS
set -euo pipefail

echo "==> Installing Python dependencies..."

# ASR: faster-whisper (runs on CPU, tiny model is ~75MB)
pip install faster-whisper

# TTS: edge-tts (no model download, uses Microsoft's API)
pip install edge-tts

# HTTP server (already installed, but ensuring)
pip install flask 2>/dev/null || true

echo "==> Downloading Whisper tiny model (first run caches it)..."
python3 -c "
from faster_whisper import WhisperModel
model = WhisperModel('tiny', device='cpu', compute_type='int8')
print('Whisper tiny loaded OK')
"

echo ""
echo "==> All set! Start with:"
echo "  python3 voice_http_server.py --no-model --port 9090 --https"
echo ""
echo "Note: LLM stage uses Ollama API (http://localhost:11434) by default."
echo "Set OLLAMA_BASE_URL env var if your Ollama is elsewhere."
