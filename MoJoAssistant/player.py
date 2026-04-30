import os
import tempfile
from typing import List, Optional
from threading import Thread
import wave
import io


class MojoAudioPlayer:
    def __init__(self):
        self._queue: List[tuple[bytes, str]] = []
        self._playing = False
        self._player_thread: Optional[Thread] = None

    def play(self, audio_bytes: bytes, format: str = "wav") -> None:
        self._queue.append((audio_bytes, format))
        if not self._playing:
            self._start_playback_loop()

    def _start_playback_loop(self) -> None:
        self._playing = True
        self._player_thread = Thread(target=self._play_loop, daemon=True)
        self._player_thread.start()

    def _play_loop(self) -> None:
        while self._queue:
            audio_bytes, fmt = self._queue.pop(0)
            try:
                if fmt == "wav":
                    self._play_wav(audio_bytes)
                else:
                    self._play_raw(audio_bytes, fmt)
            except Exception:
                pass
        self._playing = False

    def _play_wav(self, audio_bytes: bytes) -> None:
        buffer = io.BytesIO(audio_bytes)
        with wave.open(buffer, "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            frames = wav.readframes(wav.getnframes())
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(wav.getsampwidth()),
                channels=channels,
                rate=sample_rate,
                output=True,
            )
            stream.write(frames)
            stream.stop_stream()
            stream.close()
            p.terminate()
        except ImportError:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.write(fd, audio_bytes)
            os.close(fd)
            try:
                os.system(f"aplay {path}")
            finally:
                os.unlink(path)

    def _play_raw(self, audio_bytes: bytes, format: str) -> None:
        fd, path = tempfile.mkstemp(suffix=f".{format}")
        os.write(fd, audio_bytes)
        os.close(fd)
        try:
            os.system(f"aplay -f {format.upper()} {path}")
        finally:
            os.unlink(path)

    def clear_queue(self) -> None:
        self._queue.clear()

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def is_playing(self) -> bool:
        return self._playing