"""Shared checkpoint, device, metric, and evaluation behavior for FAS tools."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

from .config import LiFTConfig
from .model import LiFTClassifier
from .fas_data import CLASSES, FASTransform


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be auto, cuda, mps or cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError("Cannot write an empty result table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("class_to_idx") != CLASSES:
        raise ValueError("Unsupported FAS checkpoint format or class order")
    FASTransform(checkpoint["preprocessing"])
    return checkpoint


def load_model(path: Path, device: torch.device) -> tuple[LiFTClassifier, dict]:
    checkpoint = read_checkpoint(path)
    model = LiFTClassifier(LiFTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).float().eval(), checkpoint


def classification_metrics(labels, probabilities) -> dict:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (labels.ndim != 1 or not labels.size or probabilities.shape != labels.shape
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Expected nonempty binary labels and finite p_live values in [0,1]")
    predicted_live = probabilities >= 0.5
    fake, live = labels == 0, labels == 1
    apcer = float(predicted_live[fake].mean()) if fake.any() else None
    bpcer = float((~predicted_live[live]).mean()) if live.any() else None
    return dict(
        count=int(labels.size), fake_count=int(fake.sum()), live_count=int(live.sum()),
        threshold=0.5, accuracy=float((predicted_live == labels).mean()),
        roc_auc=float(roc_auc_score(labels, probabilities)) if fake.any() and live.any() else None,
        apcer=apcer, bpcer=bpcer, acer=(apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None,
    )


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, num_loops: int | None = None,
             *, batch_progress=None) -> tuple[dict, list[dict]]:
    model.eval()
    records, loss_sum = [], 0.0
    # Rows are true classes (fake/live), columns are predictions at the existing >= 0.5 threshold.
    confusion = np.zeros((2, 2), dtype=np.int64)
    for step, (images, labels, paths) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        logits = model(images, num_loops=num_loops).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite validation logits")
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        batch_labels = labels.cpu().tolist()
        batch_probabilities = logits.softmax(-1)[:, 1].cpu().tolist()
        for path, label, probability in zip(paths, batch_labels, batch_probabilities):
            records.append(dict(relative_path=path, label=label, p_live=probability,
                                prediction=int(probability >= 0.5)))
        if batch_progress is not None:
            for label, probability in zip(batch_labels, batch_probabilities):
                confusion[label, int(probability >= 0.5)] += 1
            counts = confusion.sum(axis=1)
            acer = ((confusion[0, 1] / counts[0] + confusion[1, 0] / counts[1]) / 2
                    if (counts > 0).all() else None)
            peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
            batch_progress(step=step, loss=loss_sum / len(records), examples=len(records),
                           accuracy=float(confusion.trace() / counts.sum()), acer=acer, peak=peak)
    metrics = classification_metrics([r["label"] for r in records], [r["p_live"] for r in records])
    metrics["loss"] = loss_sum / len(records)
    return metrics, records


def is_better(metrics: dict, best: dict | None) -> bool:
    return best is None or (metrics["acer"], metrics["loss"]) < (best["acer"], best["loss"])
