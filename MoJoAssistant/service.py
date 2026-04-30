import threading
import time
import base64
import tempfile
from typing import Optional, Callable
from dataclasses import asdict
import urllib.request
import urllib.error
import json

from .models import (
    SessionResponse,
    QueryResponse,
    PushRequest,
    PushResponse,
    PendingResponse,
    ContextResponse,
    PushType,
    ContextType,
)


class MojoVoiceService:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._running = False
        self._on_pending_audio: Optional[Callable[[bytes, str], None]] = None
        self._on_context_update: Optional[Callable[[ContextResponse], None]] = None

    def create_session(self) -> SessionResponse:
        req = urllib.request.Request(
            f"{self.base_url}/voice/session",
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
        self.session_id = data["session_id"]
        return SessionResponse(**data)

    def close_session(self) -> None:
        if not self.session_id:
            return
        req = urllib.request.Request(
            f"{self.base_url}/voice/session/{self.session_id}",
            method="DELETE",
        )
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError:
            pass
        self.session_id = None
        self.stop_polling()

    def query_audio(
        self,
        wav_bytes: bytes,
        mcp_mode: Optional[str] = None,
        role_id: Optional[str] = None,
    ) -> QueryResponse:
        if not self.session_id:
            raise RuntimeError("No active session. Call create_session() first.")
        audio_b64 = base64.b64encode(wav_bytes).decode()
        body = {"audio_base64": audio_b64}
        if mcp_mode:
            body["mcp_mode"] = mcp_mode
        if role_id:
            body["role_id"] = role_id
        req = urllib.request.Request(
            f"{self.base_url}/voice/query/{self.session_id}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
        return QueryResponse(**data)

    def push_result(self, summary: str, push_type: PushType = PushType.RESULT) -> PushResponse:
        if not self.session_id:
            raise RuntimeError("No active session. Call create_session() first.")
        body = {"type": push_type.value, "summary": summary}
        req = urllib.request.Request(
            f"{self.base_url}/voice/push/{self.session_id}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
        return PushResponse(**data)

    def poll_pending(self) -> PendingResponse:
        if not self.session_id:
            raise RuntimeError("No active session. Call create_session() first.")
        req = urllib.request.Request(
            f"{self.base_url}/voice/pending/{self.session_id}",
            method="GET",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
        return PendingResponse(**data)

    def poll_context(self) -> ContextResponse:
        if not self.session_id:
            raise RuntimeError("No active session. Call create_session() first.")
        req = urllib.request.Request(
            f"{self.base_url}/voice/context/{self.session_id}",
            method="GET",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
        return ContextResponse(**data)

    def start_polling(
        self,
        interval: float = 2.0,
        on_pending_audio: Optional[Callable[[bytes, str], None]] = None,
        on_context_update: Optional[Callable[[ContextResponse], None]] = None,
    ) -> None:
        self._on_pending_audio = on_pending_audio
        self._on_context_update = on_context_update
        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            args=(interval,),
            daemon=True,
        )
        self._poll_thread.start()

    def stop_polling(self) -> None:
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None

    def _poll_loop(self, interval: float) -> None:
        while self._running:
            try:
                pending = self.poll_pending()
                if pending.pending and pending.reply_audio_base64 and self._on_pending_audio:
                    audio_bytes = base64.b64decode(pending.reply_audio_base64)
                    self._on_pending_audio(
                        audio_bytes,
                        pending.reply_audio_format or "wav",
                    )
            except Exception:
                pass
            try:
                context = self.poll_context()
                if context.update and context.content and self._on_context_update:
                    self._on_context_update(context)
            except Exception:
                pass
            time.sleep(interval)

    def get_pending_audio_filepath(self, audio_bytes: bytes, fmt: str = "wav") -> str:
        timestamp = int(time.time() * 1000)
        fd, path = tempfile.mkstemp(suffix=f".{fmt}")
        with open(fd, "wb") as f:
            f.write(audio_bytes)
        return path