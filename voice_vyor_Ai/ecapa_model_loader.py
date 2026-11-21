from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_MODEL_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
ECAPA_MODEL_FILES = (
    "hyperparams.yaml",
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "classifier.ckpt",
    "label_encoder.txt",
)


def load_ecapa_classifier(
    encoder_classifier: Any,
    snapshot_downloader: Callable[..., str | Path],
) -> Any:
    model_directory = Path(
        snapshot_downloader(
            repo_id=ECAPA_MODEL_ID,
            revision=ECAPA_MODEL_REVISION,
            allow_patterns=list(ECAPA_MODEL_FILES),
        )
    )
    model_path = str(model_directory)
    return encoder_classifier.from_hparams(
        source=model_path,
        overrides={"pretrained_path": model_path},
    )
