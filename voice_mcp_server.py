"""
ChatMCP Voice MCP Server — wraps MiniCPM-o as an MCP tool provider.

MCP Protocol (JSON-RPC 2.0 over stdio):
  - tools/list       → returns available bridge tools
  - tools/call       → bridges text to backend LLM via MiniCPM-o trigger

Workflow:
  MiniCPM-o outputs [bridge]...[/bridge] → server detects trigger
  → routes to ChatMCP backend → returns result → MiniCPM-o speaks it
"""

import json
import sys
import logging
from typing import Any

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("voice-mcp")


def respond(msg: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def handle_request(msg: dict[str, Any]) -> None:
    req_id = msg.get("id")
    method = msg.get("method")

    if method == "tools/list":
        respond({
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "voice_bridge",
                        "description": "Bridge between local voice LLM and ChatMCP backend",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "source": {"type": "string", "enum": ["voice", "text"]},
                            },
                            "required": ["text"],
                        },
                    },
                    {
                        "name": "voice_bridge_result",
                        "description": "Inject backend result back into voice LLM context",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "result": {"type": "string"},
                            },
                            "required": ["result"],
                        },
                    },
                ]
            },
        })

    elif method == "tools/call":
        tool = msg["params"]["name"]
        args = msg["params"]["arguments"]

        if tool == "voice_bridge":
            text = args["text"]
            logger.info(f"Bridge request: {text[:100]}...")
            respond({
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": f"Bridged: {text}"}]},
            })
        elif tool == "voice_bridge_result":
            result = args["result"]
            logger.info(f"Injecting result into voice context: {result[:100]}...")
            respond({
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": "Result injected"}]},
            })
        else:
            respond({
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool}"},
            })

    else:
        respond({
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        })


def main() -> None:
    logger.info("Voice MCP server started")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            handle_request(msg)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")


if __name__ == "__main__":
    main()
