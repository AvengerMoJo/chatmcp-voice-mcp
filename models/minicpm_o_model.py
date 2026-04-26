"""
MiniCPM-o 2.6 end-to-end voice model with [bridge] trigger detection.

Model: openbmb/MiniCPM-o-2_6  (~15GB, audio→audio, single model)
Purpose: Audio in → internal Whisper+Qwen3+CosyVoice2 → audio out
         When Qwen3 generates [bridge]...[/bridge], ChatMCP intercepts.

Usage:
    model = MiniCPMoModel("openbmb/MiniCPM-o-2_6")
    model.load()
    response_wav = model.chat("input.wav")  # audio in, audio out
"""

from __future__ import annotations

import io
import logging
import re
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger("minicpm-o")

BRIDGE_PATTERN = re.compile(r"\[bridge\](.*?)\[/bridge\]", re.DOTALL)


class MiniCPMoModel:
    """MiniCPM-o 2.6: audio in → audio out, with [bridge] routing."""

    def __init__(self, model_id: str = "openbmb/MiniCPM-o-2_6"):
        self.model_id = model_id
        self.model: Optional[AutoModel] = None
        self.processor: Optional[AutoProcessor] = None
        self._bridge_callback: Optional[Callable[[str], str]] = None

    # ── Lifecycle ───────────────────────────────────────────────

    def load(self, device: str = "auto", torch_dtype: torch.dtype = torch.float16) -> None:
        logger.info(f"Loading {self.model_id} on {device}...")
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device,
            attn_implementation="sdpa",
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        logger.info(f"MiniCPM-o loaded: {sum(p.numel() for p in self.model.parameters()) / 1e9:.1f}B params")

    def unload(self) -> None:
        self.model = None
        self.processor = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ── Bridge callback ─────────────────────────────────────────

    @property
    def bridge_callback(self) -> Optional[Callable[[str], str]]:
        return self._bridge_callback

    @bridge_callback.setter
    def bridge_callback(self, cb: Callable[[str], str]) -> None:
        self._bridge_callback = cb

    # ── Core: audio in, audio out ───────────────────────────────

    def chat(self, audio: np.ndarray | str | Path, sample_rate: int = 16000) -> np.ndarray:
        """
        Main entry point: audio in → model → audio out.

        Args:
            audio: numpy array (float32, shape [samples]) or path to WAV file.
            sample_rate: sample rate if passing numpy array.

        Returns:
            numpy array (int16, shape [samples]) of the spoken response.
        """
        audio_np = self._load_audio(audio, sample_rate)

        # The model internally: Whisper encode → Qwen3 think → CosyVoice2 decode
        # We catch the text-thinking phase to intercept [bridge] triggers.
        response_audio, bridge_texts = self._generate(audio_np)
        if bridge_texts and self._bridge_callback:
            response_audio = self._handle_bridges(response_audio, bridge_texts, audio_np)

        return response_audio

    def chat_file(self, audio_path: str | Path, output_path: str | Path | None = None) -> bytes:
        """
        Convenience: WAV file in, WAV file out.
        Returns WAV bytes if output_path is None.
        """
        audio_np, sr = self._read_wav(audio_path)
        response_np = self.chat(audio_np, sr)
        wav_bytes = self._to_wav_bytes(response_np, 24000)

        if output_path:
            Path(output_path).write_bytes(wav_bytes)
        return wav_bytes

    def chat_bytes(self, wav_bytes: bytes) -> bytes:
        """WAV bytes in → WAV bytes out."""
        audio_np, sr = self._read_wav(io.BytesIO(wav_bytes))
        response_np = self.chat(audio_np, sr)
        return self._to_wav_bytes(response_np, 24000)

    # ── Internal generation ─────────────────────────────────────

    def _generate(self, audio_np: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """
        Run MiniCPM-o generation, extracting [bridge] triggers from the
        text-thinking phase before CosyVoice2 produces audio.
        """
        # MiniCPM-o chat: takes audio, returns audio + intermediate text
        # The model's chat() method handles the full pipeline internally.
        msgs = [{"role": "user", "content": [audio_np]}]

        # Run generation — the model returns both audio and text traces
        result = self.model.chat(
            msgs=msgs,
            tokenizer=self.processor,
            sampling=True,
            temperature=0.2,
            top_p=0.8,
            max_new_tokens=1024,
        )

        # Extract audio output and text trace
        response_audio = result["audio"]       # numpy int16 array
        text_trace = result.get("text", "")     # text that Qwen3 thought

        # Find [bridge] triggers in the text trace
        bridge_texts = BRIDGE_PATTERN.findall(text_trace)

        if bridge_texts:
            logger.info(f"Bridge triggers detected: {bridge_texts}")

        return response_audio, bridge_texts

    def _handle_bridges(self, audio: np.ndarray, bridge_texts: list[str], original_audio: np.ndarray) -> np.ndarray:
        """
        When [bridge] triggers are detected, route them to ChatMCP,
        inject results into context, and re-generate.
        """
        results = {}
        for text in bridge_texts:
            if self._bridge_callback:
                results[text] = self._bridge_callback(text)
            else:
                results[text] = "[no bridge callback]"

        # Build context for regeneration: original audio + bridge results
        msgs = [
            {"role": "user", "content": [original_audio]},
            {"role": "assistant", "content": f"[bridge results: {results}]"},
        ]

        try:
            result = self.model.chat(
                msgs=msgs,
                tokenizer=self.processor,
                sampling=True,
                temperature=0.2,
                top_p=0.8,
                max_new_tokens=1024,
            )
            return result["audio"]
        except Exception:
            return audio  # return original audio if regeneration fails

    # ── Audio utilities ─────────────────────────────────────────

    @staticmethod
    def _load_audio(audio: np.ndarray | str | Path, sample_rate: int = 16000) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32)
        path = Path(audio)
        if path.suffix.lower() in (".wav", ".wave"):
            data, sr = MiniCPMoModel._read_wav(path)
            return data.astype(np.float32)
        raise ValueError(f"Unsupported audio format: {path.suffix}")

    @staticmethod
    def _read_wav(path: str | Path | io.BytesIO) -> tuple[np.ndarray, int]:
        with wave.open(str(path) if isinstance(path, (str, Path)) else path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            return np.frombuffer(frames, dtype=np.int16).astype(np.float32), wf.getframerate()

    @staticmethod
    def _to_wav_bytes(audio_np: np.ndarray, sample_rate: int = 24000) -> bytes:
        if audio_np.dtype != np.int16:
            audio_np = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_np.tobytes())
        return buf.getvalue()
