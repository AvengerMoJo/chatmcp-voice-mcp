"""
Voice pipeline: audio in → ASR → LLM → TTS → audio out.

ASR:  faster-whisper (tiny model, CPU, ~75MB)
LLM:  Ollama API (configurable endpoint, any model)
TTS:  edge-tts (Microsoft online, no GPU needed, high quality)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import struct
import tempfile
import wave
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("voice-pipeline")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "en-US-JennyNeural")


class VoicePipeline:
    """End-to-end voice pipeline with real ASR/LLM/TTS."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self._whisper = None
        self._bridge_callback: Optional[Callable[[str], str]] = None

    # ── Lifecycle ───────────────────────────────────────────────

    def load(self) -> None:
        logger.info("Loading faster-whisper tiny...")
        from faster_whisper import WhisperModel
        self._whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
        logger.info("Whisper loaded")

    def unload(self) -> None:
        self._whisper = None
        logger.info("Pipeline unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._whisper is not None

    # ── Bridge ──────────────────────────────────────────────────

    @property
    def bridge_callback(self) -> Optional[Callable[[str], str]]:
        return self._bridge_callback

    @bridge_callback.setter
    def bridge_callback(self, cb: Callable[[str], str]) -> None:
        self._bridge_callback = cb

    # ── Core pipeline ───────────────────────────────────────────

    def process_audio(self, audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
        # Step 1: ASR
        asr_text = self._speech_to_text(audio_bytes, sample_rate)
        logger.info(f"ASR: {asr_text[:300]}")

        # Step 2: Bridge triggers in ASR output
        if "[bridge]" in asr_text and self._bridge_callback:
            asr_text = self._process_bridge_triggers(asr_text)

        # Step 3: LLM
        llm_text = self._text_generate(asr_text)
        logger.info(f"LLM: {llm_text[:300]}")

        # Step 4: Bridge triggers in LLM output
        if "[bridge]" in llm_text and self._bridge_callback:
            llm_text = self._process_bridge_triggers(llm_text)

        # Step 5: TTS
        response_audio = self._text_to_speech(llm_text)
        logger.info(f"TTS: {len(response_audio)} bytes")
        return response_audio

    # ── ASR: faster-whisper ─────────────────────────────────────

    def _speech_to_text(self, audio_bytes: bytes, sample_rate: int) -> str:
        import numpy as np

        # Convert bytes to float32 array
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        segments, info = self._whisper.transcribe(samples, language=None, beam_size=1)
        text = " ".join(seg.text for seg in segments)
        return text.strip() or "[silence]"

    # ── LLM: Ollama API ─────────────────────────────────────────

    def _text_generate(self, text: str) -> str:
        import urllib.request as req
        import urllib.error

        prompt = f"Respond concisely: {text}"
        body = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 256},
        }).encode()

        try:
            r = req.urlopen(
                f"{OLLAMA_BASE_URL}/api/generate",
                data=body,
                timeout=30,
            )
            result = json.loads(r.read().decode())
            return result.get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama call failed ({e}), using echo fallback")
            return f"Echo: {text[:200]}"

    # ── TTS: edge-tts ───────────────────────────────────────────

    def _text_to_speech(self, text: str) -> bytes:
        import edge_tts

        try:
            communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
            # edge-tts is async, we run it synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            audio_data = b"".join(
                chunk["data"] for chunk in loop.run_until_complete(communicate.stream())
                if chunk["type"] == "audio"
            )
            loop.close()
            if audio_data:
                return audio_data
        except Exception as e:
            logger.warning(f"edge-tts failed ({e}), using sine fallback")

        # Fallback sine wave
        duration = 1.0
        freq = 440
        num_samples = int(24000 * duration)
        samples = []
        for i in range(num_samples):
            t = i / 24000
            sample = int(16000 * 0.3 * __import__("math").sin(2 * __import__("math").pi * freq * t))
            samples.append(struct.pack("<h", sample))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"".join(samples))
        return buf.getvalue()

    # ── Bridge trigger processing ───────────────────────────────

    def _process_bridge_triggers(self, text: str) -> str:
        pattern = re.compile(r"\[bridge\](.*?)\[/bridge\]", re.DOTALL)

        def replace_match(m: re.Match) -> str:
            bridge_text = m.group(1).strip()
            if self._bridge_callback:
                result = self._bridge_callback(bridge_text)
                return f"[bridge result: {result}]"
            return "[bridge error: no callback]"

        return pattern.sub(replace_match, text)

    # ── Audio utilities ─────────────────────────────────────────

    @staticmethod
    def wav_to_pcm16(wav_bytes: bytes) -> tuple[bytes, int]:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getsampwidth() == 2, "Only 16-bit audio supported"
            return wf.readframes(wf.getnframes()), wf.getframerate()

    @staticmethod
    def pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()
