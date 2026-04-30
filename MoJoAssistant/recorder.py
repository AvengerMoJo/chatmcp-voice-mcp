import io
import wave
from typing import Optional
from record import Record, AudioFormat


class MojoRecorder:
    def __init__(
        self,
        format: AudioFormat = AudioFormat.WAV,
        channels: int = 1,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
    ):
        self.format = format
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._recorder: Optional[Record] = None
        self._buffer: Optional[io.BytesIO] = None

    def start(self) -> None:
        self._recorder = Record()
        self._buffer = io.BytesIO()
        self._recorder.stream(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            chunk_size=self.chunk_size,
        )
        self._recorder.start()

    def stop(self) -> bytes:
        if not self._recorder or not self._buffer:
            raise RuntimeError("Not recording. Call start() first.")
        self._recorder.stop()
        self._recorder.close()
        wav_bytes = self._buffer.getvalue()
        self._recorder = None
        self._buffer = None
        return wav_bytes

    @staticmethod
    def pcm_to_wav(pcm_bytes: bytes, channels: int = 1, sample_rate: int = 16000) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()

    def is_recording(self) -> bool:
        return self._recorder is not None