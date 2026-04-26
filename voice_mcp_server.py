"""
ChatMCP Voice MCP Server — wraps MiniCPM-o as an MCP tool provider.

MCP Protocol (JSON-RPC 2.0 over stdio):
  - tools/list       → returns available bridge tools
  - tools/call       → voice_bridge: send text to MiniCPM-o, detect [bridge] triggers
                       voice_bridge_result: inject backend result back

Workflow:
  User voice → MiniCPM-o → [bridge] detect → ChatMCP backend → inject result → speak
"""

from __future__ import annotations

import json
import sys
import logging
import argparse
from typing import Any

from models.minicpm_wrapper import MiniCPMWrapper, BridgeRequest

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("voice-mcp")


class VoiceMCPServer:
    """MCP server hosting MiniCPM-o with [bridge] trigger detection."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model: MiniCPMWrapper | None = None
        self._pending_bridge: BridgeRequest | None = None

    def start(self) -> None:
        self.model = MiniCPMWrapper(self.model_path)
        self.model.load()
        self.model.bridge_callback = self._on_bridge_trigger
        logger.info("Voice MCP server ready")

    def stop(self) -> None:
        if self.model:
            self.model.unload()

    # ── Bridge callback ───────────────────────────────────────

    def _on_bridge_trigger(self, text: str) -> str:
        """Called when MiniCPM-o outputs [bridge]...[/bridge]."""
        logger.info(f"Bridge triggered: {text[:200]}...")
        try:
            router = BridgeRouter(self)
            result = router.route(text)
            logger.info(f"Bridge result: {result[:200]}...")
            return result
        except Exception as e:
            logger.error(f"Bridge routing failed: {e}")
            return f"[bridge error: {e}]"

    # ── MCP request handling ──────────────────────────────────

    def handle_request(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        req_id = msg.get("id")
        method = msg.get("method")

        if method == "tools/list":
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "voice_query",
                            "description": "Process text through MiniCPM-o and detect [bridge] triggers",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "stream": {"type": "boolean"},
                                },
                                "required": ["text"],
                            },
                        },
                        {
                            "name": "voice_bridge_result",
                            "description": "Inject a backend result back into the voice model context",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "result": {"type": "string"},
                                    "request_id": {"type": "string"},
                                },
                                "required": ["result"],
                            },
                        },
                    ]
                },
            }

        if method == "tools/call":
            tool = msg["params"]["name"]
            args = msg["params"]["arguments"]
            return self._call_tool(tool, args, req_id)

        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }

    def _call_tool(self, tool: str, args: dict, req_id: Any) -> dict:
        if tool == "voice_query":
            return self._handle_voice_query(args, req_id)
        if tool == "voice_bridge_result":
            return self._handle_bridge_result(args, req_id)
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Unknown tool: {tool}"},
        }

    def _handle_voice_query(self, args: dict, req_id: Any) -> dict:
        text = args["text"]
        if self.model is None:
            return {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32000, "message": "Model not loaded"}}

        result = self.model.chat(text)
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"content": [{"type": "text", "text": result}]},
        }

    def _handle_bridge_result(self, args: dict, req_id: Any) -> dict:
        result = args["result"]
        logger.info(f"Injecting bridge result: {result[:200]}...")
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"content": [{"type": "text", "text": f"Injected: {result[:100]}..."}]},
        }


class BridgeRouter:
    """Routes [bridge] commands to appropriate handlers."""

    def __init__(self, server: VoiceMCPServer):
        self.server = server

    def route(self, text: str) -> str:
        text = text.strip()

        # Detect intent and route
        if text.startswith("get_context"):
            return self._mock("get_context", {"type": "attention"})
        if text.startswith("search"):
            return self._mock("search_memory", {"query": text})
        if text.startswith("memory"):
            return self._mock("memory", {"action": "list"})
        if text.startswith("schedule"):
            return self._mock("scheduler", {"action": "list"})

        # Default: pass through to backend
        return self._mock("chat_completion", {"text": text})

    def _mock(self, tool: str, params: dict) -> str:
        logger.info(f"Bridge routing → {tool}({params})")
        return json.dumps({"tool": tool, "params": params, "status": "routed"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/minicpm-o", help="Path to MiniCPM-o model")
    parser.add_argument("--no-model", action="store_true", help="Run without model (for testing bridge only)")
    args = parser.parse_args()

    server = VoiceMCPServer(args.model)

    if not args.no_model:
        server.start()
    else:
        logger.info("Running in bridge-only mode (no model loaded)")

    logger.info("Voice MCP server listening on stdio")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            response = server.handle_request(msg)
            if response:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")


if __name__ == "__main__":
    main()
