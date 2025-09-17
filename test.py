"""
client_invoke.py — SageMaker Runtime Inference Client
-----------------------------------------------------
Utility script to send an audio file (16 kHz mono PCM WAV) to a
SageMaker endpoint for transcription.

Steps:
  1. Load audio file and base64-encode it.
  2. Build JSON payload {"audio_b64": "<base64 string>"}.
  3. Invoke SageMaker endpoint via boto3 runtime client.
  4. Print transcript from the response.

Contract:
  • Input  (application/json): {"audio_b64": "<b64 WAV 16kHz mono PCM>"}
  • Output (application/json): {"text": "<transcript>"}
"""

import base64
import json
import logging
import boto3
from typing import Dict, Any

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def b64_encode_file(path: str) -> str:
    """
    Read a file and return its base64-encoded string.

    Args:
        path: Path to the audio file.

    Returns:
        str: Base64-encoded contents.
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------
# Main Inference Function
# ---------------------------------------------------------------------
def invoke_parakeet(
    endpoint_name: str = "nemo-parakeet-demo",
    wav_path: str = "2086-149220-0033.wav",
) -> Dict[str, Any]:
    """
    Send a 16 kHz mono WAV to a SageMaker endpoint and return transcript.

    Args:
        endpoint_name: Name of the deployed SageMaker endpoint.
        wav_path: Path to the WAV file (16kHz mono PCM).

    Returns:
        dict: JSON response with key "text".
    """
    runtime = boto3.client("sagemaker-runtime")

    log.info(f"🎧 Encoding audio file: {wav_path}")
    payload = {"audio_b64": b64_encode_file(wav_path)}

    log.info(f"🚀 Invoking SageMaker endpoint: {endpoint_name}")
    res = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )

    body = res["Body"].read().decode("utf-8")
    output = json.loads(body)

    if "text" not in output:
        raise RuntimeError(f"Unexpected response format: {output}")

    transcript = output["text"]
    log.info(f"✅ Transcription received: {transcript}")
    return output


# ---------------------------------------------------------------------
# CLI Usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    result = invoke_parakeet()
    print(f"Transcript: {result['text']}")
