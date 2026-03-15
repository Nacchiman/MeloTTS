"""LiveKit Agents TTS plugin that delegates synthesis to a MeloTTS HTTP server.

Usage
-----
Start the server (in a separate process or container)::

    uvicorn melo.tts_server:app --host 0.0.0.0 --port 8000

Then use this plugin in a LiveKit agent::

    from livekit.agents import AgentSession
    from livekit_melotts import TTS as MeloTTS

    session = AgentSession(
        tts=MeloTTS(
            base_url="http://localhost:8000",
            language="JP",
            speaker="JP-Default",
            speed=1.0,
        ),
        # ... llm, stt, etc.
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger(__name__)

NUM_CHANNELS = 1
DEFAULT_SAMPLE_RATE = 44100
MELOTTS_SERVER_TIMEOUT = 120  # seconds; TTS inference can take a while for long texts


@dataclass
class _TTSOptions:
    base_url: str
    language: str
    speaker: str
    speed: float
    sdp_ratio: float
    noise_scale: float
    noise_scale_w: float


class TTS(tts.TTS):
    """LiveKit TTS plugin backed by a MeloTTS HTTP server.

    Parameters
    ----------
    base_url:
        Base URL of the running MeloTTS server (e.g. ``"http://localhost:8000"``).
    language:
        MeloTTS language code: ``"EN"``, ``"JP"``, ``"ZH"``, ``"KR"``, ``"ES"``, or ``"FR"``.
    speaker:
        Speaker name as returned by ``GET /speakers?language=<lang>``
        (e.g. ``"EN-US"``, ``"JP-Default"``).
    speed:
        Speech speed multiplier. Default is ``1.0``.
    sample_rate:
        PCM sample rate the server will return. Must match the MeloTTS model's
        native sample rate (default ``44100``).
    sdp_ratio:
        Stochastic duration predictor mixing ratio (0 – 1). Default ``0.2``.
    noise_scale:
        Flow noise scale. Default ``0.6``.
    noise_scale_w:
        Duration noise scale. Default ``0.8``.
    http_session:
        Optional pre-existing ``aiohttp.ClientSession``. If not provided,
        LiveKit's shared HTTP session is used.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000",
        language: str = "EN",
        speaker: str = "EN-Default",
        speed: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        sdp_ratio: float = 0.2,
        noise_scale: float = 0.6,
        noise_scale_w: float = 0.8,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        self._opts = _TTSOptions(
            base_url=base_url.rstrip("/"),
            language=language,
            speaker=speaker,
            speed=speed,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
        )
        self._session = http_session

    @property
    def provider(self) -> str:
        return "MeloTTS"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def update_options(
        self,
        *,
        language: str | None = None,
        speaker: str | None = None,
        speed: float | None = None,
        sdp_ratio: float | None = None,
        noise_scale: float | None = None,
        noise_scale_w: float | None = None,
    ) -> None:
        """Update synthesis options. Changes take effect on the next call to ``synthesize``."""
        if language is not None:
            self._opts.language = language
        if speaker is not None:
            self._opts.speaker = speaker
        if speed is not None:
            self._opts.speed = speed
        if sdp_ratio is not None:
            self._opts.sdp_ratio = sdp_ratio
        if noise_scale is not None:
            self._opts.noise_scale = noise_scale
        if noise_scale_w is not None:
            self._opts.noise_scale_w = noise_scale_w


class ChunkedStream(tts.ChunkedStream):
    """Fetches PCM audio from the MeloTTS HTTP server and emits it to LiveKit."""

    def __init__(
        self,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload = {
            "text": self._input_text,
            "language": self._opts.language,
            "speaker": self._opts.speaker,
            "speed": self._opts.speed,
            "sdp_ratio": self._opts.sdp_ratio,
            "noise_scale": self._opts.noise_scale,
            "noise_scale_w": self._opts.noise_scale_w,
        }

        url = f"{self._opts.base_url}/synthesize"

        try:
            async with self._tts._ensure_session().post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=MELOTTS_SERVER_TIMEOUT,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()

                # Read the sample rate from the server response header, falling
                # back to the value the TTS instance was initialised with.
                sample_rate_header = resp.headers.get("X-Sample-Rate")
                sample_rate = (
                    int(sample_rate_header)
                    if sample_rate_header
                    else self._tts.sample_rate
                )

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                async for chunk, _ in resp.content.iter_chunks():
                    output_emitter.push(chunk)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as exc:
            raise APIStatusError(
                message=exc.message,
                status_code=exc.status,
                request_id=None,
                body=None,
            ) from exc
        except Exception as exc:
            raise APIConnectionError() from exc
