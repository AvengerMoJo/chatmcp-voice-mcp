"""
MiniCPM-o 4.5 wrapper with [bridge] trigger detection.

Architecture:
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │ Whisper-med  │───►│  Qwen3-8B    │───►│  CosyVoice2  │
  │ (audio→emb)  │    │ (text think) │    │ (text→audio) │
  └──────────────┘    └──────┬───────┘    └──────────────┘
                             │
                    ┌────────▼────────┐
                    │ [bridge] detect │──► ChatMCP backend
                    └─────────────────┘
"""

from __future__ import annotations

import json
import re
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger("minicpm_wrapper")

BRIDGE_PATTERN = re.compile(r"\[bridge\](.*?)\[/bridge\]", re.DOTALL)


@dataclass
class BridgeRequest:
    text: str
    raw_match: str
    result: Optional[str] = None
    event: threading.Event = field(default_factory=threading.Event)


class MiniCPMWrapper:
    """Wraps MiniCPM-o inference with [bridge] trigger detection."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._bridge_callback: Optional[Callable[[str], str]] = None
        self._pending: Optional[BridgeRequest] = None
        self._lock = threading.Lock()

    # ── Model lifecycle ─────────────────────────────────────────

    def load(self) -> None:
        """Load MiniCPM-o model (stub — replace with actual loading)."""
        logger.info(f"Loading MiniCPM-o from {self.model_path}...")
        # TODO: Actual model loading
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # self._model = AutoModelForCausalLM.from_pretrained(
        #     self.model_path, trust_remote_code=True, torch_dtype="auto"
        # ).eval()
        self._model = object()  # placeholder
        logger.info("MiniCPM-o loaded")

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        logger.info("MiniCPM-o unloaded")

    # ── Bridge callback ─────────────────────────────────────────

    @property
    def bridge_callback(self) -> Optional[Callable[[str], str]]:
        return self._bridge_callback

    @bridge_callback.setter
    def bridge_callback(self, cb: Callable[[str], str]) -> None:
        self._bridge_callback = cb

    # ── Inference ────────────────────────────────────────────────

    def chat(self, text: str, max_new_tokens: int = 512) -> str:
        """Run inference, detect [bridge] triggers, return final response."""
        output = self._generate(text, max_new_tokens)
        output = self._process_bridge_triggers(output)
        return output

    def chat_stream(self, text: str, max_new_tokens: int = 512):
        """Streaming inference yielding (text_chunk, is_bridge) tuples."""
        buffer = ""
        for chunk in self._generate_stream(text, max_new_tokens):
            buffer += chunk
            # Check if buffer contains a complete bridge trigger
            if "[bridge]" in buffer and "[/bridge]" in buffer:
                match = BRIDGE_PATTERN.search(buffer)
                if match:
                    bridge_text = match.group(1).strip()
                    logger.info(f"Bridge trigger detected: {bridge_text[:80]}...")
                    result = self._execute_bridge(bridge_text)
                    # Yield bridge result as a special chunk
                    yield f"[bridge result: {result}]", True
                    # Keep non-bridge text before/after the trigger
                    pre = buffer[: match.start()]
                    post = buffer[match.end() :]
                    if pre.strip():
                        yield pre, False
                    buffer = post
            yield chunk, False

        # Flush any remaining non-bridge text
        if buffer.strip():
            yield buffer, False

    # ── Internal ────────────────────────────────────────────────

    def _generate(self, text: str, max_new_tokens: int) -> str:
        """Synchronous generation (stub)."""
        # TODO: Actual model.generate() call
        return f"Mock response to: {text[:50]}..."

    def _generate_stream(self, text: str, max_new_tokens: int):
        """Streaming generation (stub — yields word by word)."""
        response = self._generate(text, max_new_tokens)
        for word in response.split(" "):
            yield word + " "

    def _process_bridge_triggers(self, text: str) -> str:
        """Find all [bridge]...[/bridge] triggers, execute them, replace with results."""
        def replace_match(m: re.Match) -> str:
            bridge_text = m.group(1).strip()
            result = self._execute_bridge(bridge_text)
            return f"[bridge result: {result}]"

        return BRIDGE_PATTERN.sub(replace_match, text)

    def _execute_bridge(self, text: str) -> str:
        """Send bridge request via callback and wait for result."""
        if self._bridge_callback is None:
            logger.warning("No bridge callback registered — returning empty")
            return ""

        with self._lock:
            req = BridgeRequest(text=text, raw_match=text)
            self._pending = req

        try:
            result = self._bridge_callback(text)
            return result
        except Exception as e:
            logger.error(f"Bridge execution failed: {e}")
            return f"[bridge error: {e}]"
        finally:
            with self._lock:
                self._pending = None

    def inject_result(self, result: str) -> None:
        """Inject a bridge result back into a pending request (async path)."""
        with self._lock:
            if self._pending is not None:
                self._pending.result = result
                self._pending.event.set()
