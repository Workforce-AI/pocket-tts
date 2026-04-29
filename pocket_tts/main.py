import base64
import io
import json
import logging
import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from queue import Queue

import torch
import typer
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from typing_extensions import Annotated

from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.default_parameters import (
    DEFAULT_AUDIO_PROMPT,
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_FRAMES_AFTER_EOS,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    MAX_TOKEN_PER_CHUNK,
    get_default_text_for_language,
)
from pocket_tts.models.tts_model import TTSModel, export_model_state
from pocket_tts.utils.logging_utils import enable_logging
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

logger = logging.getLogger(__name__)

# ------------------------------------------------------
# API key authentication
# Set POCKET_TTS_API_KEY env var to enable. If unset, auth is disabled.
# ------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_api_key_header)):
    expected = os.environ.get("POCKET_TTS_API_KEY")
    if not expected:
        return  # auth disabled — no key configured
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Set X-API-Key header.")

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool", pretty_exceptions_show_locals=False
)


# ------------------------------------------------------
# The pocket-tts server implementation
# ------------------------------------------------------

# Global model instance
tts_model: TTSModel | None = None

# In-memory voice state cache.
# Each entry: {"state": model_state_dict, "name": str}
voice_state_cache: dict[str, dict] = {}

web_app = FastAPI(
    title="Kyutai Pocket TTS API", description="Text-to-Speech generation API", version="1.0.0"
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    static_path = Path(__file__).parent / "static" / "index.html"
    content = static_path.read_text()
    print(str(tts_model.origin))
    content = content.replace(
        "DEFAULT_TEXT_PROMPT", get_default_text_for_language(str(tts_model.origin))
    )
    return content


@web_app.get("/health")
async def health():
    return {"status": "healthy"}


# ------------------------------------------------------
# Voice state helpers
# ------------------------------------------------------

def _load_voice_state_from_source(
    voice_url: str | None, voice_wav: UploadFile | None
) -> dict:
    """Compute and return a model_state from a URL or uploaded file."""
    if voice_url is not None:
        if not (
            voice_url.startswith("http://")
            or voice_url.startswith("https://")
            or voice_url.startswith("hf://")
            or voice_url in _ORIGINS_OF_PREDEFINED_VOICES
        ):
            raise HTTPException(
                status_code=400,
                detail="voice_url must start with http://, https://, hf://, or be a predefined voice name",
            )
        return tts_model._cached_get_state_for_audio_prompt(voice_url)

    if voice_wav is not None:
        suffix = Path(voice_wav.filename).suffix if voice_wav.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(voice_wav.file.read())
            tmp.flush()
            tmp_path = tmp.name
        try:
            return tts_model.get_state_for_audio_prompt(Path(tmp_path), truncate=True)
        finally:
            os.unlink(tmp_path)

    raise HTTPException(status_code=400, detail="Provide either voice_url or voice_wav")


@web_app.get("/voices", dependencies=[Depends(verify_api_key)])
async def list_voices():
    """
    List all available voices.

    Returns predefined voices (built-in catalog) and any cached custom voices.
    """
    predefined = [{"name": name, "type": "predefined"} for name in _ORIGINS_OF_PREDEFINED_VOICES]
    cached = [
        {"voice_id": vid, "name": meta["name"], "type": "cached"}
        for vid, meta in voice_state_cache.items()
    ]
    return {"predefined": predefined, "cached": cached}


@web_app.post("/voices/cache", dependencies=[Depends(verify_api_key)])
def cache_voice(
    name: str = Form(...),
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
):
    """
    Pre-compute and cache a voice state under a human-readable name.

    Call this once per avatar voice. Returns a voice_id for use in POST /tts.
    Eliminates the 5-6 second encoding cost on every TTS call.

    Args:
        name: Human-readable label for this voice (e.g. "sarah-avatar", "support-bot")
        voice_url: Predefined voice name (e.g. "alba"), or http/https/hf:// URL
        voice_wav: Uploaded WAV file (mutually exclusive with voice_url)

    Returns:
        {"voice_id": "<uuid>", "name": "<name>"}
    """
    if voice_url is not None and voice_wav is not None:
        raise HTTPException(status_code=400, detail="Provide either voice_url or voice_wav, not both")
    if voice_url is None and voice_wav is None:
        raise HTTPException(status_code=400, detail="Provide either voice_url or voice_wav")

    model_state = _load_voice_state_from_source(voice_url, voice_wav)
    voice_id = str(uuid.uuid4())
    voice_state_cache[voice_id] = {"state": model_state, "name": name}
    logger.info("Cached voice '%s' as voice_id=%s", name, voice_id)
    return {"voice_id": voice_id, "name": name}


@web_app.delete("/voices/cache/{voice_id}", dependencies=[Depends(verify_api_key)])
async def delete_cached_voice(voice_id: str):
    """Remove a cached voice to free memory."""
    if voice_id not in voice_state_cache:
        raise HTTPException(status_code=404, detail=f"voice_id '{voice_id}' not found")
    name = voice_state_cache.pop(voice_id)["name"]
    return {"deleted": voice_id, "name": name}


# ------------------------------------------------------
# Audio streaming helpers
# ------------------------------------------------------

def write_to_queue(queue, text_to_generate, model_state, max_tokens, frames_after_eos):
    """Bridges generate_audio_stream → WAV bytes via a queue for StreamingResponse."""

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue):
            self.queue = queue

        def write(self, data):
            self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    audio_chunks = tts_model.generate_audio_stream(
        model_state=model_state,
        text_to_generate=text_to_generate,
        max_tokens=max_tokens,
        frames_after_eos=frames_after_eos,
    )
    stream_audio_chunks(FileLikeToQueue(queue), audio_chunks, tts_model.config.mimi.sample_rate)


def generate_wav_stream(
    text_to_generate: str,
    model_state: dict,
    max_tokens: int,
    frames_after_eos: int | None,
    temperature: float | None,
):
    """Yields WAV bytes as they are produced."""
    original_temp = tts_model.temp
    if temperature is not None:
        tts_model.temp = temperature
    try:
        queue = Queue()
        thread = threading.Thread(
            target=write_to_queue,
            args=(queue, text_to_generate, model_state, max_tokens, frames_after_eos),
        )
        thread.start()
        while True:
            data = queue.get()
            if data is None:
                break
            yield data
        thread.join()
    finally:
        tts_model.temp = original_temp


def generate_pcm_base64_stream(
    text_to_generate: str,
    model_state: dict,
    max_tokens: int,
    frames_after_eos: int | None,
    temperature: float | None,
):
    """
    Yields newline-delimited JSON where each line contains ~1 second of audio.

    Format per line:
        {"audio": "<base64-encoded PCM 16-bit 24kHz>"}

    Compatible with LiveAvatar WebSocket requirements:
        - PCM 16-bit 24kHz
        - Base64-encoded
        - ~1 second chunks (~48 KB raw, ~64 KB base64 — well under 1 MB limit)
    """
    original_temp = tts_model.temp
    if temperature is not None:
        tts_model.temp = temperature
    try:
        sample_rate = tts_model.config.mimi.sample_rate  # 24000
        chunk_samples = sample_rate  # 1 second per chunk
        buffer = torch.zeros(0)

        for audio_chunk in tts_model.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            max_tokens=max_tokens,
            frames_after_eos=frames_after_eos,
        ):
            buffer = torch.cat([buffer, audio_chunk.cpu()])

            while buffer.shape[0] >= chunk_samples:
                chunk, buffer = buffer[:chunk_samples], buffer[chunk_samples:]
                pcm_bytes = (chunk.clamp(-1, 1) * 32767).short().numpy().tobytes()
                yield json.dumps({"audio": base64.b64encode(pcm_bytes).decode()}) + "\n"

        # flush any remaining samples (< 1 second) as a final partial chunk
        if buffer.shape[0] > 0:
            pcm_bytes = (buffer.clamp(-1, 1) * 32767).short().numpy().tobytes()
            yield json.dumps({"audio": base64.b64encode(pcm_bytes).decode()}) + "\n"
    finally:
        tts_model.temp = original_temp


# ------------------------------------------------------
# TTS endpoint
# ------------------------------------------------------

@web_app.post("/tts", dependencies=[Depends(verify_api_key)])
def text_to_speech(
    text: str = Form(...),
    voice_id: str | None = Form(None),
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
    response_format: str = Form("wav"),
    temperature: float | None = Form(None),
    frames_after_eos: int | None = Form(None),
    max_tokens: int | None = Form(None),
):
    """
    Generate speech from text.

    Voice priority: voice_id > voice_url > voice_wav > default voice.

    Args:
        text: Text to convert to speech.
        voice_id: ID returned by POST /voices/cache. Fastest path — no encoding on each call.
        voice_url: Predefined voice name (e.g. "alba") or http/https/hf:// URL.
        voice_wav: Uploaded WAV file for voice cloning.
        response_format: "wav" (default) or "pcm_base64" (for LiveAvatar).
        temperature: Sampling temperature. Higher = more expressive, less stable. Default: model setting.
        frames_after_eos: Silence padding frames after speech ends. Default: auto.
        max_tokens: Max tokens per sentence chunk. Default: 250.

    Response formats:
        wav        — audio/wav stream (chunked transfer encoding)
        pcm_base64 — newline-delimited JSON, each line: {"audio": "<base64 PCM 16-bit 24kHz>"}
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if response_format not in ("wav", "pcm_base64"):
        raise HTTPException(status_code=400, detail="response_format must be 'wav' or 'pcm_base64'")

    provided = sum(x is not None for x in [voice_id, voice_url, voice_wav])
    if provided > 1:
        raise HTTPException(status_code=400, detail="Provide at most one of: voice_id, voice_url, voice_wav")

    # Resolve voice state
    if voice_id is not None:
        if voice_id not in voice_state_cache:
            raise HTTPException(
                status_code=404,
                detail=f"voice_id '{voice_id}' not found. Call POST /voices/cache first.",
            )
        model_state = voice_state_cache[voice_id]["state"]
    elif voice_url is not None or voice_wav is not None:
        model_state = _load_voice_state_from_source(voice_url, voice_wav)
    else:
        model_state = tts_model._cached_get_state_for_audio_prompt(DEFAULT_AUDIO_PROMPT)

    effective_max_tokens = max_tokens if max_tokens is not None else MAX_TOKEN_PER_CHUNK

    if response_format == "pcm_base64":
        return StreamingResponse(
            generate_pcm_base64_stream(text, model_state, effective_max_tokens, frames_after_eos, temperature),
            media_type="application/x-ndjson",
            headers={"Transfer-Encoding": "chunked"},
        )

    return StreamingResponse(
        generate_wav_stream(text, model_state, effective_max_tokens, frames_after_eos, temperature),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


# ------------------------------------------------------
# CLI: serve
# ------------------------------------------------------

@cli_app.command()
def serve(
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help="Language for the TTS model. "
            "'english_2026-01', 'english_2026-04', 'english', 'french_24l', 'german_24l', 'portuguese', 'italian', 'spanish'."
            " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-04'.",
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to locally-saved model config .yaml file. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
    quantize: Annotated[
        bool, typer.Option(help="Apply int8 quantization to reduce memory usage")
    ] = False,
):
    """Start the FastAPI server."""

    global tts_model
    tts_model = TTSModel.load_model(language=language, config=config, quantize=quantize)

    uvicorn.run("pocket_tts.main:web_app", host=host, port=port, reload=reload)


# ------------------------------------------------------
# CLI: generate
# ------------------------------------------------------

@cli_app.command()
def generate(
    text: Annotated[str, typer.Option(help="Text to generate")] = None,
    voice: Annotated[
        str, typer.Option(help="Path to audio conditioning file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help=(
                "Language for the TTS model. "
                "'english_2026-01', 'english_2026-04', 'english', 'french_24l', 'spanish_24l',"
                "'german_24l', 'portuguese_24l', 'italian_24l'."
                " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-04'. "
                "The '24l' variants are bigger models, "
                "not distilled yet and here only as preview. They're not the final "
                "models for those languages."
            ),
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to locally-saved model config .yaml file. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Temperature for generation")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS threshold")] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    output_path: Annotated[
        str, typer.Option(help="Output path for generated audio")
    ] = "./tts_output.wav",
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
    max_tokens: Annotated[
        int, typer.Option(help="Maximum number of tokens per chunk.")
    ] = MAX_TOKEN_PER_CHUNK,
    quantize: Annotated[
        bool, typer.Option(help="Apply int8 quantization to reduce memory usage")
    ] = False,
):
    """Generate speech using Kyutai Pocket TTS."""
    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        if text is None:
            text = get_default_text_for_language(language)
        if text == "-":
            text = sys.stdin.read()

        if not text.strip():
            logger.error("No input received from stdin.")
            raise typer.Exit(code=1)
        tts_model = TTSModel.load_model(
            language=language,
            config=config,
            temp=temperature,
            lsd_decode_steps=lsd_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            quantize=quantize,
        )
        tts_model.to(device)

        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state_for_voice,
            text_to_generate=text,
            frames_after_eos=frames_after_eos,
            max_tokens=max_tokens,
        )

        stream_audio_chunks(output_path, audio_chunks, tts_model.config.mimi.sample_rate)

        if output_path != "-":
            logger.info("Results written in %s", output_path)
        logger.info("-" * 20)
        logger.info(
            "If you want to try multiple voices and prompts quickly, try the `serve` command."
        )
        logger.info(
            "If you like Kyutai projects, comment, like, subscribe at https://x.com/kyutai_labs"
        )


# ------------------------------------------------------
# CLI: export-voice
# ------------------------------------------------------

@cli_app.command()
def export_voice(
    audio_path: Annotated[
        str, typer.Argument(help="Audio file or directory to convert and export")
    ],
    export_path: Annotated[str, typer.Argument(help="Output file or directory")],
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help=(
                "Language for the TTS model. "
                "'english_2026-01', 'english_2026-04', 'english', 'french_24l', 'german_24l','spanish_24l',"
                " 'portuguese_24l', 'italian_24l'."
                " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-04'. "
                "The '24l' variants are bigger models, "
                "not distilled yet and here only as preview."
            ),
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to locally-saved model config .yaml file. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
):
    """Convert and save audio to .safetensors file"""

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(language=language, config=config)
        model_state = tts_model.get_state_for_audio_prompt(
            audio_conditioning=audio_path, truncate=True
        )
        export_model_state(model_state, export_path)


if __name__ == "__main__":
    cli_app()
