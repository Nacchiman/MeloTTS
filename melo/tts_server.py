"""FastAPI-based HTTP server that wraps MeloTTS for use as a TTS backend.

Run with:
    uvicorn melo.tts_server:app --host 0.0.0.0 --port 8000
or:
    python -m melo.tts_server
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

_UVICORN_TO_PY_LEVEL: dict[str, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": logging.DEBUG,
}

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from melo.api import TTS

logger = logging.getLogger(__name__)

# Language -> TTS model cache (lazy loaded on first request)
_models: dict[str, TTS] = {}
_model_lock = asyncio.Lock()

# Languages to pre-load and warm up at startup (populated by CLI / direct use)
_warmup_languages: list[str] = []

# Short warmup text per language — just enough to exercise the full inference path
_WARMUP_TEXT: dict[str, str] = {
    "EN": "Hello.",
    "JP": "こんにちは。",
    "ZH": "你好。",
    "KR": "안녕하세요.",
    "ES": "Hola.",
    "FR": "Bonjour.",
}
_WARMUP_TEXT_DEFAULT = "Hello."


def _warmup_sync(model: TTS) -> None:
    """Run a short dummy inference to trigger lazy init (BERT load, JIT, etc.)."""
    language = model.language
    text = _WARMUP_TEXT.get(language, _WARMUP_TEXT_DEFAULT)
    speaker_id = next(iter(model.hps.data.spk2id.values()))
    model.tts_to_file(text, speaker_id, output_path=None, quiet=True)


async def _warmup_model(language: str) -> None:
    """Load model and run a warmup inference, logging timing."""
    model = await _get_model(language)
    logger.info("[warmup] starting warmup inference for language: %s", language)
    t0 = time.perf_counter()
    await asyncio.to_thread(_warmup_sync, model)
    elapsed = time.perf_counter() - t0
    logger.info("[warmup] done for language: %s (%.3fs)", language, elapsed)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """FastAPI lifespan: warm up models at startup."""
    if _warmup_languages:
        logger.info("[warmup] warming up %d language(s): %s", len(_warmup_languages), _warmup_languages)
        await asyncio.gather(*(_warmup_model(lang) for lang in _warmup_languages))
        logger.info("[warmup] all models ready")
    yield


app = FastAPI(title="MeloTTS Server", version="1.0.0", lifespan=lifespan)


async def _get_model(language: str) -> TTS:
    """Return a cached TTS model, loading it on first access."""
    if language not in _models:
        async with _model_lock:
            # Double-checked locking after acquiring the lock
            if language not in _models:
                logger.info("Loading MeloTTS model for language: %s", language)
                model = await asyncio.to_thread(TTS, language=language)
                _models[language] = model
                logger.info("Model loaded for language: %s (sample_rate=%d)", language, model.hps.data.sampling_rate)
    return _models[language]


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "EN"
    speaker: str = "EN-Default"
    speed: float = 1.0
    sdp_ratio: float = 0.2
    noise_scale: float = 0.6
    noise_scale_w: float = 0.8


def _synthesize_sync(
    model: TTS, req: SynthesizeRequest, req_id: str
) -> tuple[np.ndarray, int]:
    """Run synchronous TTS inference. Called via asyncio.to_thread."""
    spk2id: dict[str, int] = model.hps.data.spk2id
    if req.speaker not in spk2id:
        raise ValueError(
            f"Unknown speaker '{req.speaker}' for language '{req.language}'. "
            f"Available: {list(spk2id.keys())}"
        )
    speaker_id = spk2id[req.speaker]

    char_count = len(req.text)
    logger.debug(
        "[%s] inference start | lang=%s speaker=%s speed=%.2f chars=%d text=%r",
        req_id, req.language, req.speaker, req.speed, char_count, req.text,
    )

    t0 = time.perf_counter()
    audio: np.ndarray = model.tts_to_file(
        req.text,
        speaker_id,
        output_path=None,
        sdp_ratio=req.sdp_ratio,
        noise_scale=req.noise_scale,
        noise_scale_w=req.noise_scale_w,
        speed=req.speed,
        quiet=True,
    )
    elapsed = time.perf_counter() - t0

    sample_rate = model.hps.data.sampling_rate
    audio_duration = len(audio) / sample_rate
    rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")

    logger.debug(
        "[%s] inference done  | elapsed=%.3fs audio_duration=%.3fs RTF=%.3f samples=%d",
        req_id, elapsed, audio_duration, rtf, len(audio),
    )

    return audio, sample_rate


@app.post(
    "/synthesize",
    response_class=Response,
    responses={200: {"content": {"audio/pcm": {}}}},
    summary="Synthesize text to PCM audio",
    description=(
        "Converts text to 16-bit signed mono PCM audio. "
        "The sample rate is returned in the X-Sample-Rate response header."
    ),
)
async def synthesize(req: SynthesizeRequest) -> Response:
    """Convert text to int16 PCM audio and return it as audio/pcm."""
    req_id = uuid.uuid4().hex[:8]
    t_request = time.perf_counter()

    logger.debug(
        "[%s] request received | lang=%s speaker=%s speed=%.2f chars=%d",
        req_id, req.language, req.speaker, req.speed, len(req.text),
    )

    try:
        model = await _get_model(req.language)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {exc}") from exc

    try:
        audio, sample_rate = await asyncio.to_thread(_synthesize_sync, model, req, req_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[%s] TTS inference failed", req_id)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    # Convert float32 [-1, 1] to int16 PCM
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    pcm_bytes = audio_int16.tobytes()

    total_elapsed = time.perf_counter() - t_request
    audio_duration = len(audio) / sample_rate
    logger.debug(
        "[%s] response ready   | total_elapsed=%.3fs pcm_bytes=%d sample_rate=%d audio_duration=%.3fs",
        req_id, total_elapsed, len(pcm_bytes), sample_rate, audio_duration,
    )

    return Response(
        content=pcm_bytes,
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(sample_rate)},
    )


@app.get(
    "/speakers",
    summary="List available speakers for a language",
)
async def list_speakers(language: str = Query("EN", description="Language code")) -> dict[str, Any]:
    """Return a mapping of speaker name -> speaker ID for the given language."""
    try:
        model = await _get_model(language)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {exc}") from exc
    return dict(model.hps.data.spk2id)


@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--host", "-h", default="0.0.0.0", show_default=True, help="Bind host")
    @click.option("--port", "-p", default=8000, show_default=True, type=int, help="Bind port")
    @click.option("--language", "-l", multiple=True, help="Pre-load model(s) at startup (e.g. -l EN -l JP)")
    @click.option("--log-level", default="info", show_default=True, help="Uvicorn log level")
    def main(host: str, port: int, language: tuple[str, ...], log_level: str) -> None:
        """Start the MeloTTS HTTP server."""
        # Configure the Python root logger to match the requested log level.
        # uvicorn's --log-level only affects uvicorn's own loggers; without this
        # the application logger stays at WARNING and emits nothing.
        py_level = _UVICORN_TO_PY_LEVEL.get(log_level.lower(), logging.INFO)
        logging.basicConfig(
            level=py_level,
            format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        if language:
            _warmup_languages.extend(language)

        uvicorn.run(app, host=host, port=port, log_level=log_level)

    main()
