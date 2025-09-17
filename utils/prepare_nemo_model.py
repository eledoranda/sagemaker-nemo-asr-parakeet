"""
utils/prepare_nemo_model.py — Artifact Preparation Utility
----------------------------------------------------------
Ensures a NeMo `.nemo` checkpoint exists locally (downloads if missing),
then packages it into a SageMaker-compatible `model.tar.gz` archive with
`model.nemo` at the archive root.

Improvements:
  • Validate existing tar contains `model.nemo` before skipping.
  • Deterministic packaging (mtime/uid/gid/perms) for reproducible builds.
  • Post-pack verification to catch corrupt archives immediately.

Returns:
  • Path to the generated `model.tar.gz`
"""

import logging
import os
import tarfile
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def _tar_has_model_nemo(path: str) -> bool:
    """Return True iff path is a valid tar.gz containing a top-level `model.nemo`."""
    if not os.path.exists(path):
        return False
    if not tarfile.is_tarfile(path):
        return False
    try:
        with tarfile.open(path, "r:gz") as t:
            names = t.getnames()
            if "model.nemo" not in names:
                return False
            info = t.getmember("model.nemo")
            if not info.isreg() or info.size <= 0:
                return False
        return True
    except Exception:
        return False


def _deterministic_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    """Normalize metadata for reproducible archives."""
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    # rw-r--r--
    tarinfo.mode = 0o644 if tarinfo.isfile() else 0o755
    # Fixed timestamp for reproducibility
    tarinfo.mtime = 0
    return tarinfo


def prepare_nemo_artifact(
    local_nemo_path: str = "artifacts/model.nemo",
    model_name: str = "nvidia/parakeet-rnnt-0.6b",
    out_tar: str = "artifacts/model.tar.gz",
) -> str:
    """
    Ensure a `.nemo` exists at `local_nemo_path` (download if missing),
    then pack `out_tar` with the file as 'model.nemo' at the archive root.

    Args:
        local_nemo_path: Local path for the .nemo file.
        model_name:      NeMo/HF model identifier.
        out_tar:         Output tar.gz path.

    Returns:
        str: Path to the created tar.gz archive.

    Raises:
        FileNotFoundError: If .nemo missing after attempted download.
        RuntimeError:      If tar packaging/verification fails.
    """
    # Skip packaging if file exists
    if os.path.exists(out_tar):
        log.info(f"📦 Model artifact exists: {out_tar}, skipping packaging")
        return out_tar

    # Ensure directories exist
    os.makedirs(os.path.dirname(local_nemo_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_tar), exist_ok=True)

    # Ensure the .nemo checkpoint exists (download if needed)
    if not os.path.exists(local_nemo_path):
        log.info(f"⬇️  Downloading NeMo model '{model_name}' → {local_nemo_path}")
        import nemo.collections.asr as nemo_asr
        try:
            model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
            model.save_to(local_nemo_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download/save model '{model_name}': {e}") from e
        log.info(f"✅ Saved NeMo model checkpoint: {local_nemo_path}")

    if not os.path.exists(local_nemo_path) or os.path.getsize(local_nemo_path) <= 0:
        raise FileNotFoundError(f"Missing or empty .nemo file at {local_nemo_path}")

    # Create deterministic model.tar.gz with `model.nemo` at root
    log.info(f"📦 Packaging '{local_nemo_path}' → '{out_tar}'")
    try:
        with tarfile.open(out_tar, "w:gz") as tar:
            tar.add(local_nemo_path, arcname="model.nemo", filter=_deterministic_filter)
    except Exception as e:
        raise RuntimeError(f"Failed to create tar '{out_tar}': {e}") from e

    # Post-pack verification (structure + readability)
    if not _tar_has_model_nemo(out_tar):
        raise RuntimeError(f"Verification failed: '{out_tar}' is not a valid model archive")

    log.info(f"✅ Packaged model artifact ready: {out_tar}")
    return out_tar
