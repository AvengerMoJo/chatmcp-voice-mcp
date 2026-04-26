#!/usr/bin/env bash
# Test script for ChatMCP Voice HTTP Server
# Usage: bash test_voice_server.sh [server_url]
# Default: http://localhost:9090

set -euo pipefail

BASE="${1:-http://localhost:9090}"

echo "================================================"
echo " Voice MCP Server Test"
echo " Server: $BASE"
echo "================================================"

echo ""
echo "--- 1. Health Check ---"
curl -s "$BASE/health" | python3 -m json.tool
echo ""

echo "--- 2. List Tools ---"
curl -s "$BASE/tools" | python3 -m json.tool
echo ""

echo "--- 3. Chat Query ---"
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"text":"What is the project deadline?"}' \
  | python3 -m json.tool
echo ""

echo "--- 4. Chat with [bridge] trigger ---"
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"text":"Can you [bridge]search_memory(\"deadline\")[/bridge] for me?"}' \
  | python3 -m json.tool
echo ""

echo "--- 5. Bridge Result Injection ---"
curl -s -X POST "$BASE/bridge" \
  -H "Content-Type: application/json" \
  -d '{"result":"The deadline is next Friday"}' \
  | python3 -m json.tool
echo ""

echo "--- 6. Invalid Path ---"
curl -s "$BASE/notfound" | python3 -m json.tool
echo ""

echo "================================================"
echo " All tests completed"
echo "================================================"
