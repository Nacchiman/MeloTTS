"""
Backchannel TTS generator: Gradio UI with direct MeloTTS inference.
Phrases are loaded from a JSON file (folder_name -> text). Output ZIP uses
folder_name/001.wav, 002.wav, ... with generation-time IDs.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import click
import gradio as gr
import librosa
import numpy as np
import soundfile as sf

from melo.api import TTS

# ---------------------------------------------------------------------------
# Config and defaults (env overrides)
# ---------------------------------------------------------------------------
DEFAULT_PHRASES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "backchannel_phrases.json",
)
LANGUAGES = ["EN", "ES", "FR", "ZH", "JP", "KR"]
MAX_PATTERNS = 10
DEFAULT_PATTERNS = 5

_models: dict[str, TTS] = {}


def _get_model(language: str) -> TTS:
    if language not in _models:
        _models[language] = TTS(language=language, device="auto")
    return _models[language]


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v is not None else default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v is not None else default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Synthesis and resampling
# ---------------------------------------------------------------------------
def _resample_if_needed(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return librosa.resample(
        y=audio.astype(np.float32),
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type="kaiser_best",
    )


def synthesize_one(
    language: str,
    speaker: str,
    text: str,
    speed: float,
    sample_rate_out: int,
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
) -> tuple[int, np.ndarray]:
    """Run TTS for one phrase. Returns (sample_rate, audio numpy float32)."""
    model = _get_model(language)
    spk2id = model.hps.data.spk2id
    if speaker not in spk2id:
        raise ValueError(
            f"Unknown speaker '{speaker}' for language '{language}'. "
            f"Available: {list(spk2id.keys())}"
        )
    speaker_id = spk2id[speaker]
    model_sr = model.hps.data.sampling_rate
    audio = model.tts_to_file(
        text,
        speaker_id,
        output_path=None,
        sdp_ratio=sdp_ratio,
        noise_scale=noise_scale,
        noise_scale_w=noise_scale_w,
        speed=speed,
        quiet=True,
    )
    audio = _resample_if_needed(audio, model_sr, sample_rate_out)
    return sample_rate_out, audio


def _audio_to_gradio(audio: np.ndarray, sr: int):
    """Return value suitable for gr.Audio: (sample_rate, numpy)."""
    return (sr, audio)


def _safe_folder_name(phrase_key: str) -> str:
    """Sanitize phrase key for use as filesystem folder name."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in phrase_key).strip() or "phrase"


def _load_phrase_audios_from_disk(temp_root: str | None, phrase_key: str) -> list:
    """Load WAVs for one phrase from temp_root/folder/001.wav, 002.wav, ... Return list of MAX_PATTERNS (sr, audio) or None."""
    out = [None] * MAX_PATTERNS
    if not temp_root or not phrase_key or not os.path.isdir(temp_root):
        return out
    folder = os.path.join(temp_root, _safe_folder_name(phrase_key))
    if not os.path.isdir(folder):
        return out
    for i in range(MAX_PATTERNS):
        wav_path = os.path.join(folder, f"{i + 1:03d}.wav")
        if os.path.isfile(wav_path):
            try:
                audio, sr = sf.read(wav_path, dtype="float32")
                out[i] = (sr, audio)
            except Exception:
                pass
    return out


# ---------------------------------------------------------------------------
# Load phrases
# ---------------------------------------------------------------------------
def load_phrases(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON must be a dictionary (folder_name -> text)")
    return {k: v for k, v in data.items() if isinstance(v, str) and not k.startswith("_")}


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------
def load_speakers(language: str):
    model = _get_model(language)
    choices = list(model.hps.data.spk2id.keys())
    default = _env_str("MELOTTS_SPEAKER", "EN-US")
    if default not in choices:
        default = choices[0]
    return gr.update(value=default, choices=choices)


def generate_all_phrases(
    n_str: str,
    language: str,
    speaker: str,
    speed: float,
    sample_rate: int,
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    phrases: dict[str, str],
    first_phrase_key: str | None,
    progress=gr.Progress(),
):
    """
    Step 1: Generate all phrases × n patterns; write to temp dir (folder_name/001.wav, ...).
    Returns (temp_root, aud_0, ..., aud_9) for the first phrase so Confirm view can show them.
    """
    try:
        n = min(max(1, int(n_str)), MAX_PATTERNS)
    except (ValueError, TypeError):
        n = DEFAULT_PATTERNS
    tmpdir = tempfile.mkdtemp(prefix="backchannel_dataset_")
    items = list(phrases.items())
    for folder_name, text in progress.tqdm(items, desc="Step 1: Generating all"):
        folder_path = os.path.join(tmpdir, _safe_folder_name(folder_name))
        os.makedirs(folder_path, exist_ok=True)
        for i in range(n):
            sr, audio = synthesize_one(
                language, speaker, text, speed, sample_rate,
                sdp_ratio, noise_scale, noise_scale_w,
            )
            wav_path = os.path.join(folder_path, f"{i + 1:03d}.wav")
            sf.write(wav_path, audio, sr)
    audios = _load_phrase_audios_from_disk(tmpdir, first_phrase_key or (items[0][0] if items else ""))
    return (tmpdir,) + tuple(audios)


def regenerate_slot_to_disk(
    temp_root: str | None,
    phrase_key: str | None,
    slot_index: int,
    language: str,
    speaker: str,
    speed: float,
    sample_rate: int,
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    phrases: dict[str, str],
):
    """
    Step 2: Regenerate one slot; overwrite file in temp_root and return (sr, audio) for that slot.
    """
    if not temp_root or not phrase_key or phrase_key not in phrases:
        return None
    text = phrases[phrase_key]
    sr, audio = synthesize_one(
        language, speaker, text, speed, sample_rate,
        sdp_ratio, noise_scale, noise_scale_w,
    )
    folder_path = os.path.join(temp_root, _safe_folder_name(phrase_key))
    os.makedirs(folder_path, exist_ok=True)
    wav_path = os.path.join(folder_path, f"{slot_index + 1:03d}.wav")
    sf.write(wav_path, audio, sr)
    return _audio_to_gradio(audio, sr)


def build_zip_from_temp(temp_root: str | None):
    """
    Step 3: Zip the current dataset at temp_root; return path to zip, or None if no dataset.
    """
    if not temp_root or not os.path.isdir(temp_root):
        return None
    zip_path = os.path.join(tempfile.gettempdir(), f"backchannel_{os.path.basename(temp_root)}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(temp_root):
            for f in files:
                abs_path = os.path.join(root, f)
                arcname = os.path.relpath(abs_path, temp_root)
                zf.write(abs_path, arcname)
    return zip_path


def build_bulk_zip(
    n_str: str,
    language: str,
    speaker: str,
    speed: float,
    sample_rate: int,
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    phrases: dict[str, str],
    progress=gr.Progress(),
):
    """Generate all phrases × n patterns, write to folder_name/001.wav, ..., then zip. Returns path to zip."""
    try:
        n = min(max(1, int(n_str)), MAX_PATTERNS)
    except (ValueError, TypeError):
        n = DEFAULT_PATTERNS
    tmpdir = tempfile.mkdtemp(prefix="backchannel_zip_")
    zip_path = os.path.join(tmpdir, "backchannel.zip")
    items = list(phrases.items())
    for idx, (folder_name, text) in enumerate(progress.tqdm(items, desc="Phrases")):
        # Sanitize folder name for filesystem
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in folder_name).strip() or "phrase"
        folder_path = os.path.join(tmpdir, safe_name)
        os.makedirs(folder_path, exist_ok=True)
        for i in range(n):
            sr, audio = synthesize_one(
                language, speaker, text, speed, sample_rate,
                sdp_ratio, noise_scale, noise_scale_w,
            )
            wav_name = f"{i + 1:03d}.wav"
            wav_path = os.path.join(folder_path, wav_name)
            sf.write(wav_path, audio, sr)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(tmpdir):
            for f in files:
                abs_path = os.path.join(root, f)
                if os.path.normpath(abs_path) == os.path.normpath(zip_path):
                    continue
                arcname = os.path.relpath(abs_path, tmpdir)
                zf.write(abs_path, arcname)
    return zip_path


# ---------------------------------------------------------------------------
# Gradio UI (3-step: Generate all -> Confirm -> Download)
# ---------------------------------------------------------------------------
def create_demo(phrases_path: str):
    phrases = load_phrases(phrases_path)
    phrase_keys = sorted(phrases.keys())

    default_language = _env_str("MELOTTS_LANGUAGE", "EN")
    default_speaker = _env_str("MELOTTS_SPEAKER", "EN-US")
    default_speed = _env_float("MELOTTS_SPEED", 1.0)
    default_sr = _env_int("MELOTTS_SAMPLE_RATE", 44100)
    default_sdp = _env_float("MELOTTS_SDP_RATIO", 0.2)
    default_noise = _env_float("MELOTTS_NOISE_SCALE", 0.6)
    default_noise_w = _env_float("MELOTTS_NOISE_SCALE_W", 0.8)

    with gr.Blocks(title="Backchannel TTS") as demo:
        gr.Markdown(
            "# Backchannel TTS Generator\n\n"
            "**3 steps:** (1) Generate all phrases × n patterns → (2) Confirm: browse and re-generate any slot → (3) Download ZIP of the dataset."
        )

        generated_root = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                language = gr.Radio(LANGUAGES, value=default_language, label="Language")
                speaker = gr.Dropdown(value=default_speaker, label="Speaker", allow_custom_value=False)
                speed = gr.Slider(0.1, 10.0, value=default_speed, step=0.1, label="Speed")
                sample_rate = gr.Number(value=default_sr, precision=0, label="Sample rate (output)")
                sdp_ratio = gr.Slider(0.0, 1.0, value=default_sdp, step=0.05, label="SDP ratio")
                noise_scale = gr.Slider(0.0, 2.0, value=default_noise, step=0.05, label="Noise scale")
                noise_scale_w = gr.Slider(0.0, 2.0, value=default_noise_w, step=0.05, label="Noise scale W")
                language.change(load_speakers, inputs=[language], outputs=[speaker])

        gr.Markdown("### Step 1: Generate all")
        with gr.Row():
            n_patterns = gr.Number(
                value=DEFAULT_PATTERNS, precision=0, minimum=1, maximum=MAX_PATTERNS,
                label="Patterns per phrase",
            )
            gen_all_btn = gr.Button("Generate all phrases", variant="primary")
        step1_status = gr.Markdown(value="", visible=True)

        gr.Markdown("### Step 2: Confirm — browse and re-generate")
        phrase_dropdown = gr.Dropdown(
            choices=phrase_keys,
            value=phrase_keys[0] if phrase_keys else None,
            label="Phrase (folder name)",
        )
        audio_outputs = []
        regen_buttons = []
        for i in range(MAX_PATTERNS):
            with gr.Row():
                aud = gr.Audio(label=f"Pattern {i + 1}", type="numpy", interactive=False)
                btn = gr.Button(f"Regenerate {i + 1}", size="sm")
            audio_outputs.append(aud)
            regen_buttons.append(btn)

        gr.Markdown("### Step 3: Download dataset")
        download_btn = gr.Button("Download ZIP", variant="secondary")
        zip_download = gr.DownloadButton(label="Download ZIP", visible=False)

        common_inputs = [
            language, speaker, speed, sample_rate, sdp_ratio, noise_scale, noise_scale_w
        ]

        def on_generate_all(n, lang, spk, spd, sr, sdp, nsc, nscw, first_key):
            result = generate_all_phrases(
                n, lang, spk, spd, sr, sdp, nsc, nscw, phrases, first_key
            )
            root = result[0]
            audios = result[1 : 1 + MAX_PATTERNS]
            status = "Done. Go to Step 2 to confirm and Step 3 to download." if root else ""
            return root, *audios, gr.update(value=status)

        gen_all_btn.click(
            fn=on_generate_all,
            inputs=[n_patterns] + common_inputs + [phrase_dropdown],
            outputs=[generated_root] + audio_outputs + [step1_status],
        )

        def on_phrase_change(root, key):
            audios = _load_phrase_audios_from_disk(root, key or "")
            return tuple(audios)

        phrase_dropdown.change(
            fn=on_phrase_change,
            inputs=[generated_root, phrase_dropdown],
            outputs=audio_outputs,
        )

        for i in range(MAX_PATTERNS):
            def _regen(temp_root, key, *params, idx=i):
                return regenerate_slot_to_disk(
                    temp_root, key, idx, *params, phrases=phrases
                )

            regen_buttons[i].click(
                fn=_regen,
                inputs=[generated_root, phrase_dropdown] + common_inputs,
                outputs=[audio_outputs[i]],
            )

        def on_download(root):
            path = build_zip_from_temp(root)
            if path:
                return gr.update(visible=True, value=path)
            return gr.update(visible=False)

        download_btn.click(
            fn=on_download,
            inputs=[generated_root],
            outputs=[zip_download],
        )

        demo.load(
            fn=lambda: load_speakers(default_language),
            inputs=[],
            outputs=[speaker],
        )

    return demo


@click.command()
@click.option("--phrases", "-p", default=None, envvar="BACKCHANNEL_PHRASES_JSON", help="Path to phrases JSON (folder_name -> text)")
@click.option("--share", "-s", is_flag=True, default=False, help="Create public share link")
@click.option("--host", "-h", default=None, help="Server host")
@click.option("--port", "-P", type=int, default=7860, help="Server port")
def main(phrases: str | None, share: bool, host: str | None, port: int):
    phrases_path = phrases or DEFAULT_PHRASES_PATH
    if not os.path.isfile(phrases_path):
        raise SystemExit(f"Phrases file not found: {phrases_path}")
    demo = create_demo(phrases_path)
    demo.queue(api_open=False).launch(share=share, server_name=host, server_port=port)


if __name__ == "__main__":
    main()
