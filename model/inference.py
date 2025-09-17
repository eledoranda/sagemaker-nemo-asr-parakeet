"""
inference.py — Minimal & strict SageMaker inference handler
-----------------------------------------------------------
Contract:
  • Input  (application/json): {"audio_b64": "<base64-encoded WAV 16 kHz mono PCM>"}
  • Output (application/json): {"text": "<ASR transcript>"}

Notes:
  • WAV MUST be 16,000 Hz, mono, PCM. No resampling or channel mixing is performed.
  • Intentional strictness: invalid format → 4XX error raised by the container.
  • Sample code for demonstration; hardens logging and validation.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any, Tuple

import soundfile as sf
import torch

# ---------------------------------------------------------------------
# Globals / Config
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR_REQUIRED = 16_000  # Hz


# ---------------------------------------------------------------------
# SageMaker Model Loading
# ---------------------------------------------------------------------
def model_fn(model_dir: str) -> Any:
    """
    Load the NeMo ASR model once per container.
    Expects a 'model.nemo' artifact in model_dir.
    """
    ckpt = os.path.join(model_dir, "model.nemo")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing model artifact: {ckpt}")
    
    # Skip packaging if file already exists
    logger.info(f"Model file found: {ckpt}, skipping packaging")
    
    # Lazy import to keep container import cost minimal during packaging
    import nemo.collections.asr as nemo_asr

    logger.info(f"Loading NeMo model from: {ckpt} (device={DEVICE})")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(ckpt)
    model = model.to(DEVICE).eval()

    # Keep CPU usage contained when running on CPU
    torch.set_num_threads(1)
    # Optional: disable autotune variability
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False

    return model


# ---------------------------------------------------------------------
# Request Parsing
# ---------------------------------------------------------------------
def input_fn(request_body: str, content_type: str = "application/json") -> "np.ndarray":
    """
    Parse the JSON request, decode base64 WAV, and return a float32 mono array.
    Strict format: 16 kHz, mono, PCM. No resampling performed.
    """
    if content_type != "application/json":
        raise ValueError("Unsupported content type. Only 'application/json' is accepted.")

    try:
        payload = json.loads(request_body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if "audio_b64" not in payload:
        raise ValueError("Missing required field 'audio_b64' in JSON payload.")

    b64 = payload["audio_b64"]
    if not isinstance(b64, str) or not b64:
        raise ValueError("'audio_b64' must be a non-empty base64 string.")

    # Decode base64 → bytes
    try:
        wav_bytes = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 in 'audio_b64': {e}") from e

    # Decode WAV via soundfile
    try:
        # always_2d=True → shape (T, C)
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    except Exception as e:
        raise ValueError(f"Failed to decode WAV: {e}") from e

    # Strict checks (no automatic fixes)
    if sr != SR_REQUIRED:
        raise ValueError(f"Expected sample rate {SR_REQUIRED} Hz, but received {sr} Hz.")
    if audio.shape[1] != 1:
        raise ValueError("Expected mono WAV (1 channel).")

    # Flatten (T, 1) → (T,)
    audio = audio[:, 0]

    # Optional sanity check: non-empty audio
    if audio.size == 0:
        raise ValueError("Decoded audio is empty.")

    return audio


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
def predict_fn(audio_np, model) -> str:
    """
    Run transcription using the loaded NeMo model.
    NeMo expects a list of NumPy arrays.
    """
    import numpy as np

    # Ensure contiguous float32 array
    if not isinstance(audio_np, np.ndarray):
        import numpy as np  # local import safety
        audio_np = np.asarray(audio_np, dtype="float32")
    else:
        audio_np = audio_np.astype("float32", copy=False)

    with torch.inference_mode():
        # Transcribe returns List[str]; we take index 0 for single input
        text = model.transcribe([audio_np])[0]
    return text


# ---------------------------------------------------------------------
# Response Formatting
# ---------------------------------------------------------------------
def output_fn(pred: str, accept: str = "application/json") -> Tuple[str, str]:
    """
    Wrap transcript into JSON.
    """
    if accept != "application/json":
        # SageMaker container will still send it, but we advertise JSON only
        logger.warning("Non-standard 'accept' requested; returning application/json anyway.")

    body = json.dumps({"text": pred}, ensure_ascii=False)
    return body, "application/json"
