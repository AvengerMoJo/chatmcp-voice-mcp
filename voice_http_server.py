"""
HTTP wrapper for the Voice MCP server — curl-friendly testing endpoint.

Usage:
  python voice_http_server.py --port 9090 --no-model
  curl http://localhost:9090/chat -d '{"text":"Hello"}'
  curl http://localhost:9090/tools
"""

from __future__ import annotations

import json
import logging
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from voice_mcp_server import VoiceMCPServer

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("voice-http")


class VoiceHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler exposing voice MCP functionality."""

    server: VoiceMCPServer  # set by main()

    def do_GET(self) -> None:
        if self.path == "/tools":
            self._respond({"tools": ["voice_query", "voice_bridge_result"]})
        elif self.path == "/health":
            self._respond({"status": "ok", "model_loaded": self.server.model is not None})
        else:
            self._respond({"error": "not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._respond({"error": "invalid JSON"}, 400)
            return

        if self.path == "/chat":
            text = data.get("text", "")
            result = self.server._handle_voice_query({"text": text}, None)
            self._respond(result.get("result", result))
        elif self.path == "/bridge":
            result = data.get("result", "")
            r = self.server._handle_bridge_result({"result": result}, None)
            self._respond(r.get("result", r))
        else:
            self._respond({"error": "not found"}, 404)

    def _respond(self, data: dict | list | str, status: int = 200) -> None:
        self.send_response(status)
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
    args = parser.parse_args()

    server = VoiceMCPServer("./models/minicpm-o")
    if not args.no_model:
        server.start()
    else:
        logger.info("Bridge-only mode (no model)")

    VoiceHTTPHandler.server = server
    httpd = HTTPServer((args.host, args.port), VoiceHTTPHandler)
    logger.info(f"HTTP server listening on http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
