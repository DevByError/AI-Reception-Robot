"""
Stage 04 - the final words of each question through Groq's cloud Whisper.

    STT_CLOUD=groq          in .env, together with
    GROQ_API_KEY=...        a free key from https://console.groq.com (the same key the answers use)

Why: the robot's own Whisper is small.en (244M parameters) so that it runs on a laptop. Groq runs
Whisper large-v3 (1.55B) on its own hardware in a fraction of a second, and the bigger model is
much better with accents, names and a noisy room. On Groq's free plan that is 20 requests a
minute and 2,000 a day - one request per question, far more than a reception desk needs.

What stays on the laptop: everything that has to be instant. The partial transcripts while the
visitor talks, the end-of-turn decision and the voice gate all keep using the local recognizer;
only the FINISHED question is sent. The local recognizer transcribes the same audio at the same
time, so if Groq is slow (STT_CLOUD_TIMEOUT_S), down, out of quota or the key is wrong, the local
transcript is used and the visitor never notices. A failing Groq is left alone for a while
(longer for a bad key) instead of being retried on every question.

Privacy: with this on, a recording of each question leaves the robot (docs/PRIVACY.md).

Whisper's known weakness is inventing words on audio with no speech in it ("Thank you."). A
cloud transcript is therefore only used when the local recognizer also heard words, or when
Whisper itself is confident the clip was speech; and when large-v3 is itself unsure of what it
heard, the local transcript (with its own confidence report) is used instead.
"""

from __future__ import annotations

import io
import logging
import math
import re
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

log = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_TRANSCRIBE_URL = f"{GROQ_BASE_URL}/audio/transcriptions"
GROQ_MODELS_URL = f"{GROQ_BASE_URL}/models"

# How long a failing cloud recognizer is left alone before it is tried again (seconds).
REST_AFTER_BAD_KEY_S = 600.0  # 401/403: nothing will change until somebody fixes .env
REST_AFTER_QUOTA_S = 60.0  # 429: the per-minute limit resets within a minute
REST_AFTER_FAILURE_S = 30.0  # timeouts, network errors, 5xx
# A cloud transcript of a clip the local recognizer found no words in is only trusted when
# Whisper's own "this is not speech" probability is below this.
MAX_NO_SPEECH_WITHOUT_LOCAL_WORDS = 0.3
# Whisper's own "this decode failed" line: an average log-probability under -1.0 is where
# Whisper retries at a higher temperature. Below it the cloud text is a guess, and the local
# transcript (with its per-word confidence) is the better thing to act on.
MIN_CLOUD_AVG_PROB = math.exp(-1.0)
# Whisper's prompt is limited to 224 tokens; stay well under it.
MAX_PROMPT_CHARS = 500

_KEY_PATTERN = re.compile(r"gsk_[0-9A-Za-z]{4,}")


class CloudSTTError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


def _short(text, limit: int = 160) -> str:
    """One line, short, and never an API key (an error body can echo a request)."""
    return _KEY_PATTERN.sub("gsk_...", " ".join(str(text).split()))[:limit]


class GroqTranscriber:
    """One HTTPS request per finished question to Groq's OpenAI-compatible transcription API.
    The connection is kept open between questions, so only the first one pays for the TLS
    handshake."""

    name = "Groq"

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3",
        timeout_s: float = 3.0,
        sample_rate: int = 16000,
        http=None,
    ):
        if not api_key:
            raise CloudSTTError("no GROQ_API_KEY in .env")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.sample_rate = sample_rate
        self._http = http  # tests hand in a stand-in with .post/.get; else a requests.Session

    def _client(self):
        if self._http is None:
            import requests

            self._http = requests.Session()
        return self._http

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def check(self) -> None:
        """Startup check: is the key accepted and is the model offered? Lists the models, which
        sends no audio and uses none of the audio quota. Raises CloudSTTError."""
        try:
            resp = self._client().get(GROQ_MODELS_URL, headers=self._headers(), timeout=10)
        except Exception as exc:  # noqa: BLE001
            raise CloudSTTError(f"{type(exc).__name__}: {_short(exc)}") from exc
        if resp.status_code != 200:
            raise CloudSTTError(f"HTTP {resp.status_code}: {_short(resp.text)}", resp.status_code)
        try:
            ids = {m.get("id") for m in resp.json().get("data", [])}
        except Exception:  # noqa: BLE001 - an odd listing is not a reason to stop using it
            return
        if ids and self.model not in ids:
            raise CloudSTTError(
                f"Groq does not offer the model '{self.model}' - check " f"GROQ_STT_MODEL in .env",
                404,
            )

    def transcribe(
        self, pcm16: bytes, language: str = "en", prompt: str | None = None
    ) -> tuple[str, float | None, float]:
        """(text, average token probability or None, no-speech probability).
        Raises CloudSTTError."""
        data = {
            "model": self.model,
            "language": language,
            "temperature": "0",
            "response_format": "verbose_json",
        }
        if prompt:
            data["prompt"] = prompt
        try:
            resp = self._client().post(
                GROQ_TRANSCRIBE_URL,
                headers=self._headers(),
                files={"file": ("question.wav", _wav_bytes(pcm16, self.sample_rate), "audio/wav")},
                data=data,
                timeout=self.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - timeouts, DNS, no network: all "unavailable"
            raise CloudSTTError(f"{type(exc).__name__}: {_short(exc)}") from exc
        if resp.status_code != 200:
            raise CloudSTTError(f"HTTP {resp.status_code}: {_short(resp.text)}", resp.status_code)
        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise CloudSTTError(f"unreadable answer: {_short(resp.text)}") from exc
        text = (body.get("text") or "").strip()
        segments = body.get("segments") or []
        logprobs = [s["avg_logprob"] for s in segments if s.get("avg_logprob") is not None]
        no_speech = [s["no_speech_prob"] for s in segments if s.get("no_speech_prob") is not None]
        avg_prob = math.exp(sum(logprobs) / len(logprobs)) if logprobs else None
        return text, avg_prob, (sum(no_speech) / len(no_speech) if no_speech else 0.0)


def _has_words(text: str) -> bool:
    cleaned = text.replace("[??]", " ")
    return any(ch.isalpha() for ch in cleaned)


class CloudFinalEngine:
    """Same protocol as WhisperSTTEngine: partial previews from the local engine, the finished
    question from the cloud with the local transcript as the backup. Anything else (hotwords,
    the device it runs on) is the local engine's."""

    def __init__(
        self,
        local,
        cloud,
        make_confidence,
        make_chunk,
        log_transcripts: bool = True,
        clock=time.monotonic,
    ):
        self.local = local
        self.cloud = cloud
        self._make_confidence = make_confidence
        self._make_chunk = make_chunk
        self._log_transcripts = log_transcripts
        self._clock = clock
        self._resting_until = 0.0
        # A slow request is abandoned at the deadline but finishes in the background, so leave
        # room for a couple of those next to the current one.
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="stt-cloud")
        self.deadline_s = float(getattr(cloud, "timeout_s", 3.0))
        self.last_confidence = None
        self.last_source = "local"

    def __getattr__(self, name):
        # only called for attributes this wrapper does not have itself
        return getattr(self.__dict__["local"], name)

    def transcribe_stream(self, audio_chunks, locale):
        """Partial previews: local, instant, never sent anywhere."""
        chunks = self.local.transcribe_stream(audio_chunks, locale)
        self.last_confidence = getattr(self.local, "last_confidence", None)
        self.last_source = "local"
        return chunks

    def _prompt(self) -> str | None:
        prompt = getattr(self.local, "initial_prompt", None) or ""
        hotwords = getattr(self.local, "hotwords", None) or ""
        if hotwords and len(prompt) < MAX_PROMPT_CHARS - 20:
            room = MAX_PROMPT_CHARS - len(prompt) - 1
            words = hotwords if len(hotwords) <= room else hotwords[:room].rsplit(",", 1)[0]
            prompt = f"{prompt} {words}".strip()
        return prompt[:MAX_PROMPT_CHARS] or None

    def _rest(self, exc: CloudSTTError) -> None:
        rest = (
            REST_AFTER_BAD_KEY_S
            if exc.status in (401, 403, 404)
            else REST_AFTER_QUOTA_S if exc.status == 429 else REST_AFTER_FAILURE_S
        )
        self._resting_until = self._clock() + rest
        hint = " - check GROQ_API_KEY in .env" if exc.status in (401, 403) else ""
        log.warning(
            f"[stt] {self.cloud.name} could not transcribe ({exc}){hint}. Using the local "
            f"Whisper; trying {self.cloud.name} again in {rest:.0f}s."
        )

    def check_in_background(self) -> None:
        """Says at startup whether the key works, without holding up startup. A rejected key
        puts the cloud to rest at once, so the first visitor does not wait for it to fail."""

        def _run():
            try:
                self.cloud.check()
            except CloudSTTError as exc:
                self._rest(exc)
                return
            log.info(
                f"[stt] {self.cloud.name} {self.cloud.model} is ready - finished questions are "
                f"transcribed there, with the local Whisper as the backup."
            )

        threading.Thread(target=_run, name="stt-cloud-check", daemon=True).start()

    def transcribe_final(self, audio_chunks, locale):
        if self._clock() < self._resting_until:
            return self.transcribe_stream(audio_chunks, locale)
        started = self._clock()
        cloud_job = self._pool.submit(
            self.cloud.transcribe, b"".join(audio_chunks), locale.split("-")[0], self._prompt()
        )
        # The local transcript runs here, at the same time, on the thread that already does the
        # partials - so the GPU is never used from two threads at once.
        local_chunks = self.transcribe_stream(audio_chunks, locale)
        local_text = local_chunks[-1].text if local_chunks else ""
        try:
            wait = max(0.05, self.deadline_s - (self._clock() - started))
            text, avg_prob, no_speech = cloud_job.result(timeout=wait)
        except FutureTimeout:
            self._rest(CloudSTTError(f"no answer within {self.deadline_s:.1f}s"))
            return local_chunks
        except CloudSTTError as exc:
            self._rest(exc)
            return local_chunks
        except Exception as exc:  # noqa: BLE001 - never lose the question to a cloud bug
            self._rest(CloudSTTError(f"{type(exc).__name__}: {_short(exc)}"))
            return local_chunks
        took_ms = (self._clock() - started) * 1000

        if not text or (
            not _has_words(local_text) and no_speech >= MAX_NO_SPEECH_WITHOUT_LOCAL_WORDS
        ):
            # Nothing, or words on audio the local recognizer heard no words in and Whisper
            # itself doubts is speech - the classic invented "Thank you."
            return local_chunks
        if avg_prob is not None and avg_prob < MIN_CLOUD_AVG_PROB and _has_words(local_text):
            log.info(f"[stt] {self.cloud.name} was unsure of this one - using the local Whisper.")
            return local_chunks
        if self._log_transcripts:
            log.info(
                f'[stt] {self.cloud.name} ({self.cloud.model}, {took_ms:.0f}ms): "{text}" '
                f'- the local Whisper heard "{local_text}"'
            )
        else:
            log.info(
                f"[stt] {self.cloud.name} ({self.cloud.model}) transcribed it in {took_ms:.0f}ms"
            )
        self.last_confidence = self._make_confidence(
            total_word_count=len(text.split()),
            low_confidence_word_count=0,
            corrected_word_count=0,
            avg_word_probability=avg_prob,
            no_speech_prob=no_speech,
        )
        self.last_source = self.cloud.name
        return [self._make_chunk(text=text, is_final=True)]
