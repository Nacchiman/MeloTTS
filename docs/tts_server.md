## MeloTTS HTTP Server & LiveKit Agent Plugin

### Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Starting the TTS Server](#starting-the-tts-server)
  - [Basic startup](#basic-startup)
  - [CLI options](#cli-options)
  - [Pre-loading models at startup](#pre-loading-models-at-startup)
- [API Reference](#api-reference)
  - [POST /synthesize](#post-synthesize)
  - [GET /speakers](#get-speakers)
  - [GET /health](#get-health)
- [LiveKit Agent Integration](#livekit-agent-integration)
  - [Installation](#installation)
  - [Plugin usage](#plugin-usage)
  - [Plugin options](#plugin-options)
  - [Switching speakers mid-conversation](#switching-speakers-mid-conversation)
- [Architecture](#architecture)

---

### Overview

`melo/tts_server.py` is a [FastAPI](https://fastapi.tiangolo.com/) HTTP server that wraps MeloTTS and exposes a simple REST API for speech synthesis.  
`livekit_melotts/` is a [LiveKit Agents](https://docs.livekit.io/agents/) TTS plugin that connects to this server, allowing you to use MeloTTS as the TTS backend in a LiveKit voice agent.

---

### Prerequisites

Install the extra dependencies required by the server and the plugin:

```bash
pip install fastapi "uvicorn[standard]" aiohttp livekit-agents
```

MeloTTS itself must already be installed (see [install.md](install.md)).

---

### Starting the TTS Server

#### Basic startup

```bash
uvicorn melo.tts_server:app --host 0.0.0.0 --port 8000
```

Or use the built-in CLI entry point:

```bash
python -m melo.tts_server --host 0.0.0.0 --port 8000
```

The interactive API documentation (Swagger UI) is available at:  
`http://localhost:8000/docs`

#### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` / `-h` | `0.0.0.0` | Bind address |
| `--port` / `-p` | `8000` | Bind port |
| `--language` / `-l` | *(none)* | Pre-load model(s) at startup. Can be repeated. |
| `--log-level` | `info` | Uvicorn log level (`debug`, `info`, `warning`, `error`) |

#### Pre-loading models at startup

By default, models are loaded on the first synthesis request for each language (lazy loading).  
To pre-load specific languages and avoid the first-request delay, pass `--language`:

```bash
python -m melo.tts_server --language EN --language JP
```

---

### API Reference

#### POST /synthesize

Synthesize text and return raw 16-bit signed mono PCM audio.

**Request body (JSON):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *(required)* | Text to synthesize |
| `language` | string | `"EN"` | Language code: `EN`, `JP`, `ZH`, `KR`, `ES`, `FR` |
| `speaker` | string | `"EN-Default"` | Speaker name (see [GET /speakers](#get-speakers)) |
| `speed` | float | `1.0` | Speech speed multiplier |
| `sdp_ratio` | float | `0.2` | Stochastic duration predictor mixing ratio (0–1) |
| `noise_scale` | float | `0.6` | Flow noise scale |
| `noise_scale_w` | float | `0.8` | Duration noise scale |

**Example request:**

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "こんにちは、世界！",
    "language": "JP",
    "speaker": "JP-Default",
    "speed": 1.0
  }' \
  --output output.pcm
```

**Response:**

- Content-Type: `audio/pcm`
- Body: raw int16 PCM audio (mono)
- Header `X-Sample-Rate`: native sample rate of the model (e.g. `44100`)

To convert the raw PCM to a WAV file using FFmpeg:

```bash
ffmpeg -f s16le -ar 44100 -ac 1 -i output.pcm output.wav
```

---

#### GET /speakers

Return available speaker names and their integer IDs for a given language.

**Query parameter:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `language` | `EN` | Language code |

**Example:**

```bash
curl "http://localhost:8000/speakers?language=EN"
```

```json
{
  "EN-Default": 0,
  "EN-US": 1,
  "EN-BR": 2,
  "EN_INDIA": 3,
  "EN-AU": 4
}
```

---

#### GET /health

Returns `{"status": "ok"}` when the server is running.

```bash
curl http://localhost:8000/health
```

---

### LiveKit Agent Integration

#### Installation

The plugin lives in the `livekit_melotts/` directory of this repository.  
Install it as an editable package alongside MeloTTS:

```bash
pip install -e .               # MeloTTS
pip install livekit-agents aiohttp
```

The `livekit_melotts` package is importable directly from the repository root.

#### Plugin usage

```python
from livekit.agents import AgentSession, Agent
from livekit_melotts import TTS as MeloTTS

class MyAgent(Agent):
    ...

session = AgentSession(
    tts=MeloTTS(
        base_url="http://localhost:8000",
        language="JP",
        speaker="JP-Default",
        speed=1.0,
    ),
    # ... llm, stt, vad, etc.
)
```

#### Plugin options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | str | `"http://localhost:8000"` | URL of the MeloTTS HTTP server |
| `language` | str | `"EN"` | Language code |
| `speaker` | str | `"EN-Default"` | Speaker name |
| `speed` | float | `1.0` | Speech speed multiplier |
| `sample_rate` | int | `44100` | PCM sample rate (must match the server's output) |
| `sdp_ratio` | float | `0.2` | Stochastic duration predictor ratio |
| `noise_scale` | float | `0.6` | Flow noise scale |
| `noise_scale_w` | float | `0.8` | Duration noise scale |
| `http_session` | `aiohttp.ClientSession \| None` | `None` | Custom HTTP session (optional) |

#### Switching speakers mid-conversation

```python
tts = MeloTTS(base_url="http://localhost:8000", language="EN", speaker="EN-US")

# Later, switch to a different speaker:
tts.update_options(speaker="EN-BR", speed=1.1)
```

---

### Architecture

```
LiveKit Agent (AgentSession)
        │
        │  HTTP POST /synthesize  (aiohttp, async)
        ▼
MeloTTS HTTP Server  (FastAPI + Uvicorn)
  melo/tts_server.py
        │
        │  asyncio.to_thread → TTS.tts_to_file()
        ▼
MeloTTS Model  (melo/api.py)
  float32 numpy array
        │
        │  clip → int16 PCM bytes
        ▼
HTTP Response  audio/pcm  +  X-Sample-Rate header
        │
        ▼
ChunkedStream._run()  →  output_emitter.push(chunk)
        │
        ▼
LiveKit AudioFrame pipeline
```

Models are **lazy-loaded** per language on the first synthesis request and cached for subsequent requests.  
Because `TTS.tts_to_file()` is a synchronous blocking call (PyTorch inference), it is offloaded to a thread pool via `asyncio.to_thread()` to keep the server's event loop responsive.
