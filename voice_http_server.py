"""
HTTP server for ChatMCP Voice — audio upload, text chat, bridge.

Endpoints:
  GET   /health           – server status
  GET   /tools            – list tools
  POST  /chat             – text in, text out (for testing)
  POST  /voice_upload     – WAV file in, WAV file out (full pipeline)
  POST  /bridge           – inject backend result

Usage:
  python voice_http_server.py --port 9090 --no-model
  curl -X POST http://localhost:9090/chat -d '{"text":"Hello"}'
  curl -X POST http://localhost:9090/voice_upload -F "audio=@test.wav" -o response.wav
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from voice_mcp_server import VoiceMCPServer
from voice_pipeline import VoicePipeline

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("voice-http")


class VoiceHTTPHandler(BaseHTTPRequestHandler):
    server: VoiceMCPServer  # set by main()
    pipeline: VoicePipeline  # set by main()

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self) -> None:
        vs: VoiceMCPServer = VoiceHTTPHandler.server
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/tools":
            self._respond({"tools": ["voice_query", "voice_bridge_result", "voice_upload"]})
        elif self.path == "/health":
            loaded = vs.model is not None
            pipeline_loaded = VoiceHTTPHandler.pipeline.is_loaded
            self._respond({"status": "ok", "model_loaded": loaded, "pipeline_loaded": pipeline_loaded})
        else:
            self._respond({"error": "not found"}, 404)

    def _serve_html(self) -> None:
        html_path = Path(__file__).parent / "voice_test.html"
        if not html_path.exists():
            self._respond({"error": "voice_test.html not found"}, 404)
            return
        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        with open(html_path, "rb") as f:
            self.wfile.write(f.read())

    def do_POST(self) -> None:
        vs: VoiceMCPServer = VoiceHTTPHandler.server

        if self.path == "/voice_upload":
            self._handle_voice_upload()
            return

        # JSON endpoints
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._respond({"error": "invalid JSON"}, 400)
            return

        if self.path == "/chat":
            text = data.get("text", "")
            rpc = vs._handle_voice_query({"text": text}, None)
            content = rpc.get("result", {}).get("content", [])
            self._respond({"response": content[0]["text"] if content else "no response"})
        elif self.path == "/bridge":
            result_text = data.get("result", "")
            rpc = vs._handle_bridge_result({"result": result_text}, None)
            content = rpc.get("result", {}).get("content", [])
            self._respond({"response": content[0]["text"] if content else "injected"})
        else:
            self._respond({"error": "not found"}, 404)

    # ── Voice upload ────────────────────────────────────────────

    def _handle_voice_upload(self) -> None:
        """Accept multipart WAV upload, return WAV response."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._respond({"error": "multipart/form-data required"}, 400)
            return

        # Parse multipart — extract boundary and read audio field
        boundary = content_type.split("boundary=")[-1].strip()
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)

        try:
            audio_bytes = self._extract_multipart_audio(raw, boundary)
        except ValueError as e:
            self._respond({"error": str(e)}, 400)
            return

        # Process through voice pipeline
        pipeline = VoiceHTTPHandler.pipeline
        try:
            # Convert whatever format the browser sent to PCM16 WAV
            wav_bytes = self._to_wav(audio_bytes)
            pcm16, sr = pipeline.wav_to_pcm16(wav_bytes)
            response_wav = pipeline.process_audio(pcm16, sr)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._respond({"error": f"pipeline error: {e}"}, 500)
            return

        # Return WAV response
        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Disposition", 'attachment; filename="response.wav"')
        self.end_headers()
        self.wfile.write(response_wav)

    @staticmethod
    def _to_wav(raw_bytes: bytes) -> bytes:
        """Convert any audio format to WAV using ffmpeg, or return as-is if already WAV."""
        if raw_bytes[:4] == b"RIFF" and b"WAVE" in raw_bytes[:12]:
            return raw_bytes  # already WAV
        try:
            import subprocess as sp
            p = sp.run(
                ["ffmpeg", "-y", "-i", "pipe:0", "-f", "wav", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", "pipe:1"],
                input=raw_bytes, capture_output=True, timeout=15,
            )
            if p.returncode == 0 and len(p.stdout) > 44:
                return p.stdout
        except FileNotFoundError:
            logger.warning("ffmpeg not found, returning raw bytes (may fail)")
        except Exception as e:
            logger.warning(f"ffmpeg conversion failed: {e}")
        return raw_bytes

    @staticmethod
    def _extract_multipart_audio(raw: bytes, boundary: str) -> bytes:
        """Extract the first audio field from multipart data."""
        boundary_bytes = f"--{boundary}".encode()
        parts = raw.split(boundary_bytes)
        for part in parts:
            if b"Content-Disposition" not in part:
                continue
            # Find blank line separating headers from body
            header_end = part.find(b"\r\n\r\n")
            if header_end == -1:
                continue
            body = part[header_end + 4:]
            # Strip trailing \r\n--\r\n
            body = body.strip().rstrip(b"-").strip()
            if len(body) > 44:  # minimum valid WAV header
                return body
        raise ValueError("No audio data found in upload")

    # ── Response helpers ────────────────────────────────────────

    def _respond(self, data: Any, status: int = 200) -> None:
        self.send_response(status)
        self._send_cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info(f"HTTP: {fmt % args}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--model", default="./models/minicpm-o")
    parser.add_argument("--https", action="store_true", help="Enable HTTPS with self-signed cert")
    parser.add_argument("--cert-file", default=None, help="Path to SSL cert (auto-generated if not set)")
    parser.add_argument("--key-file", default=None, help="Path to SSL key (auto-generated if not set)")
    args = parser.parse_args()

    mcp_server = VoiceMCPServer(args.model)
    if not args.no_model:
        mcp_server.start()
    else:
        logger.info("Bridge-only mode (no model)")

    pipeline = VoicePipeline(args.model)
    if not args.no_model:
        pipeline.load()
    else:
        logger.info("Pipeline in stub mode")

    VoiceHTTPHandler.server = mcp_server
    VoiceHTTPHandler.pipeline = pipeline
    httpd = HTTPServer((args.host, args.port), VoiceHTTPHandler)
    protocol = "http"

    if args.https or args.cert_file:
        import ssl
        cert_file = args.cert_file
        key_file = args.key_file
        if not cert_file or not key_file:
            # Auto-generate self-signed cert
            import tempfile, subprocess
            cert_dir = Path(tempfile.gettempdir()) / ".voice-mcp-certs"
            cert_dir.mkdir(exist_ok=True)
            cert_file = str(cert_dir / "cert.pem")
            key_file = str(cert_dir / "key.pem")
            if not Path(cert_file).exists():
                subprocess.run([
                    "openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", key_file,
                    "-out", cert_file, "-days", "365", "-nodes",
                    "-subj", "/CN=localhost"
                ], capture_output=True)
                logger.info(f"Generated self-signed cert at {cert_file}")

        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(cert_file, key_file)
        httpd.socket = ssl_ctx.wrap_socket(httpd.socket, server_side=True)
        protocol = "https"

    logger.info(f"{protocol.upper()} server listening on {protocol}://{args.host}:{args.port}")
    logger.info(f"  Open {protocol}://{args.host}:{args.port}  — Web UI (voice recorder)")
    logger.info(f"  Open http://{args.host}:{args.port}  — Web UI (voice recorder)")
    logger.info(f"  POST /chat          — text in, text out")
    logger.info(f"  POST /voice_upload   — WAV in, WAV out (full pipeline)")
    logger.info(f"  POST /bridge         — inject backend result")
    logger.info(f"  GET  /health         — server status")
    logger.info(f"  GET  /tools          — list tools")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
