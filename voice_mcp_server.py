"""
ChatMCP Voice MCP Server — wraps MiniCPM-o 2.6 end-to-end.

MCP Protocol (JSON-RPC 2.0 over stdio):
  - tools/list       → available voice tools
  - tools/call       → voice_query (audio in, audio out)
                       voice_bridge_result (inject backend result)

Model: MiniCPM-o 2.6 (openbmb/MiniCPM-o-2_6)
  Audio in → Whisper encoder → Qwen3 text think → CosyVoice2 → audio out
  During Qwen3 text-thinking, [bridge]...[/bridge] triggers ChatMCP routing.
"""

from __future__ import annotations

import json
import sys
import logging
import argparse
from pathlib import Path
from typing import Any

from models.minicpm_o_model import MiniCPMoModel

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("voice-mcp")


class VoiceMCPServer:
    """MCP server hosting MiniCPM-o with [bridge] trigger detection."""

    def __init__(self, model_path: str = "openbmb/MiniCPM-o-2_6"):
        self.model_path = model_path
        self.model: MiniCPMoModel | None = None

    def start(self) -> None:
        import torch
        self.model = MiniCPMoModel(self.model_path)
        self.model.load(device="cuda" if torch.cuda.is_available() else "cpu")
        self.model.bridge_callback = self._on_bridge_trigger
        logger.info("Voice MCP server ready (MiniCPM-o 2.6)")

    def stop(self) -> None:
        if self.model:
            self.model.unload()

    # ── Bridge callback ───────────────────────────────────────

    def _on_bridge_trigger(self, text: str) -> str:
        logger.info(f"Bridge triggered: {text[:200]}...")
        try:
            router = BridgeRouter(self)
            return router.route(text)
        except Exception as e:
            logger.error(f"Bridge routing failed: {e}")
            return f"[bridge error: {e}]"

    # ── MCP request handling ──────────────────────────────────

    def handle_request(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        req_id = msg.get("id")
        method = msg.get("method")

        if method == "tools/list":
            return self._list_tools(req_id)
        if method == "tools/call":
            tool = msg["params"]["name"]
            args = msg["params"]["arguments"]
            return self._call_tool(tool, args, req_id)

        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }

    def _list_tools(self, req_id: Any) -> dict:
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "voice_query_text",
                        "description": "Process text through MiniCPM-o and detect [bridge] triggers",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                    {
                        "name": "voice_query_audio",
                        "description": "Process speech audio through MiniCPM-o end-to-end (speech in, speech out)",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "audio_path": {"type": "string"},
                                "audio_base64": {"type": "string"},
                            },
                        },
                    },
                    {
                        "name": "voice_bridge_result",
                        "description": "Inject backend result back into voice model context",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"result": {"type": "string"}},
                            "required": ["result"],
                        },
                    },
                ]
            },
        }

    def _call_tool(self, tool: str, args: dict, req_id: Any) -> dict:
        if tool == "voice_query_text":
            return self._handle_text_query(args, req_id)
        if tool == "voice_query_audio":
            return self._handle_audio_query(args, req_id)
        if tool == "voice_bridge_result":
            return self._handle_bridge_result(args, req_id)
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Unknown tool: {tool}"},
        }

    def _handle_text_query(self, args: dict, req_id: Any) -> dict:
        text = args["text"]
        if self.model is None:
            result = f"[MiniCPM-o not loaded] Text: {text[:200]}"
        else:
            result = f"[MiniCPM-o text mode] Response to: {text[:200]}"

        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"content": [{"type": "text", "text": result}]},
        }

    def _handle_audio_query(self, args: dict, req_id: Any) -> dict:
        if self.model is None:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32000, "message": "MiniCPM-o model not loaded"},
            }

        try:
            import numpy as np

            if "audio_base64" in args:
                import base64
                import io as io_mod
                raw = base64.b64decode(args["audio_base64"])
                audio_np, sr = MiniCPMoModel._read_wav(io_mod.BytesIO(raw))
            elif "audio_path" in args:
                audio_np, sr = MiniCPMoModel._read_wav(args["audio_path"])
            else:
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32602, "message": "audio_path or audio_base64 required"},
                }

            response_audio = self.model.chat(audio_np, sr)
            wav_bytes = MiniCPMoModel._to_wav_bytes(response_audio, 24000)
            audio_b64 = __import__("base64").b64encode(wav_bytes).decode()

            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": "Audio response generated"},
                        {"type": "audio", "format": "wav", "data": audio_b64},
                    ]
                },
            }
        except Exception as e:
            logger.error(f"Audio query failed: {e}")
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32001, "message": str(e)},
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

        if text.startswith("get_context"):
            return self._result("get_context")
        if text.startswith("search"):
            return self._result("search_memory", {"query": text})
        if text.startswith("memory"):
            return self._result("memory")
        if text.startswith("schedule"):
            return self._result("scheduler")
        return self._result("chat_completion", {"text": text})

    def _result(self, tool: str, params: dict | None = None) -> str:
        return json.dumps({
            "tool": tool,
            "params": params or {},
            "status": "routed_to_chatmcp",
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openbmb/MiniCPM-o-2_6", help="Model ID or path")
    parser.add_argument("--no-model", action="store_true", help="Run without model")
    parser.add_argument("--no-loader", action="store_true", help="Skip torch import")
    args = parser.parse_args()

    server = VoiceMCPServer(args.model)

    if not args.no_model:
        server.start()
    else:
        logger.info("Bridge-only mode (MiniCPM-o not loaded)")

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
