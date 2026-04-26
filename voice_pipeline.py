"""
Voice pipeline: audio in → MiniCPM-o → audio out.
Wraps ASR (Whisper), LLM (Qwen3-8B), TTS (CosyVoice2) into one interface.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("voice-pipeline")


class VoicePipeline:
    """End-to-end voice pipeline with [bridge] trigger support."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self._loaded = False
        self._bridge_callback: Optional[Callable[[str], str]] = None

    # ── Lifecycle ───────────────────────────────────────────────

    def load(self) -> None:
        """Load all sub-models. TODO: replace stubs with real loading."""
        logger.info(f"Loading voice pipeline from {self.model_path}...")

        # TODO: Load Whisper-medium for ASR
        # self.asr = whisper.load_model("medium")

        # TODO: Load Qwen3-8B for LLM
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # self.llm = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).eval()

        # TODO: Load CosyVoice2 for TTS
        # self.tts = CosyVoice2(...)

        self._loaded = True
        logger.info("Voice pipeline loaded")

    def unload(self) -> None:
        self._loaded = False
        logger.info("Voice pipeline unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Bridge ──────────────────────────────────────────────────

    @property
    def bridge_callback(self) -> Optional[Callable[[str], str]]:
        return self._bridge_callback

    @bridge_callback.setter
    def bridge_callback(self, cb: Callable[[str], str]) -> None:
        self._bridge_callback = cb

    # ── Core pipeline ───────────────────────────────────────────

    def process_audio(self, audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
        """
        Full pipeline: audio in → text → [bridge] → text → audio out.

        Args:
            audio_bytes: Raw PCM16 mono audio data.
            sample_rate: Sample rate of input audio.

        Returns:
            Raw PCM16 mono audio data of the response.
        """
        # Step 1: ASR (speech-to-text)
        asr_text = self._speech_to_text(audio_bytes, sample_rate)
        logger.info(f"ASR: {asr_text[:200]}")

        # Step 2: Detect [bridge] triggers
        if "[bridge]" in asr_text and self._bridge_callback:
            asr_text = self._process_bridge_triggers(asr_text)

        # Step 3: LLM (text thinking)
        llm_text = self._text_generate(asr_text)
        logger.info(f"LLM: {llm_text[:200]}")

        # Step 4: Detect [bridge] in LLM output
        if "[bridge]" in llm_text and self._bridge_callback:
            llm_text = self._process_bridge_triggers(llm_text)

        # Step 5: TTS (text-to-speech)
        response_audio = self._text_to_speech(llm_text)
        logger.info(f"TTS: {len(response_audio)} bytes generated")

        return response_audio

    # ── Stub methods (replace with real model calls) ────────────

    def _speech_to_text(self, audio_bytes: bytes, sample_rate: int) -> str:
        """ASR stub. TODO: replace with Whisper-medium."""
        import hashlib
        h = hashlib.md5(audio_bytes[:4096]).hexdigest()[:8]
        logger.info(f"[STUB] ASR — audio={len(audio_bytes)}bytes, hash={h}")
        return f"[ASR stub] Audio received ({len(audio_bytes)} bytes at {sample_rate}Hz)"

    def _text_generate(self, text: str) -> str:
        """LLM stub. TODO: replace with Qwen3-8B."""
        logger.info(f"[STUB] LLM generate — text={text[:100]}...")
        return f"Response to: {text[:200]}"

    def _text_to_speech(self, text: str) -> bytes:
        """TTS stub. TODO: replace with CosyVoice2."""
        logger.info(f"[STUB] TTS — text={text[:100]}...")
        # Generate a dummy sine wave so there's actual audio to play
        import math
        import struct
        sample_rate = 24000
        duration = 1.0
        freq = 440
        num_samples = int(sample_rate * duration)
        samples = []
        for i in range(num_samples):
            t = i / sample_rate
            sample = int(16000 * math.sin(2 * math.pi * freq * t))
            samples.append(struct.pack('<h', sample))
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(samples))
        return buf.getvalue()

    # ── Bridge trigger processing ───────────────────────────────

    def _process_bridge_triggers(self, text: str) -> str:
        import re
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
        """Convert WAV bytes to raw PCM16 + sample rate."""
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            assert wf.getsampwidth() == 2, "Only 16-bit audio supported"
            return wf.readframes(wf.getnframes()), wf.getframerate()

    @staticmethod
    def pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
        """Wrap raw PCM16 bytes in a WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()
