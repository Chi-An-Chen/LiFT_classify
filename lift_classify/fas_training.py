"""Single-device FAS training with epoch-boundary resume and fixed-threshold selection."""

from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import PIL
import sklearn
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from rich.console import Console

from .config import LiFTConfig, validate_integer
from .model import LiFTClassifier
from .fas_data import AUGMENTATION, CLASSES, FASTransform, FASDataset, audit_dataset, preprocessing_spec
from .fas_progress import BatchProgress, make_progress
from .fas_runtime import (evaluate, is_better, load_model, read_checkpoint, resolve_device,
                          save_checkpoint, write_csv, write_json)


TRAIN_DEFAULTS = dict(epochs=100, batch_size=128, learning_rate=3e-4, min_learning_rate=3e-6,
                      warmup_epochs=5, weight_decay=0.05, gradient_clip=1.0, seed=42,
                      num_workers=4, precision="bf16")


def validate_config(config: dict) -> dict:
    if set(config) != {"model", "training"}:
        raise ValueError("Config must contain model and training objects")
    model = LiFTConfig(**config["model"])
    if model.in_channels != 3 or model.num_classes != 2 or model.image_size[0] != model.image_size[1]:
        raise ValueError("FAS requires square input geometry, three channels, and two classes")
    if set(config["training"]) - TRAIN_DEFAULTS.keys():
        raise ValueError("Unknown training configuration field")
    train = TRAIN_DEFAULTS | config["training"]
    for key in ("epochs", "batch_size"):
        validate_integer(train[key], key)
    for key in ("warmup_epochs", "seed", "num_workers"):
        validate_integer(train[key], key, minimum=0)
    if train["warmup_epochs"] >= train["epochs"]:
        raise ValueError("warmup_epochs must be less than epochs")
    for key in ("learning_rate", "min_learning_rate", "weight_decay", "gradient_clip"):
        value = train[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if train["learning_rate"] <= 0 or train["gradient_clip"] <= 0:
        raise ValueError("learning_rate and gradient_clip must be positive")
    if train["min_learning_rate"] > train["learning_rate"] or train["precision"] not in {"bf16", "fp32"}:
        raise ValueError("Invalid minimum LR or precision")
    # JSON-normalized config makes checkpoint comparisons independent of tuple/list input.
    return json.loads(json.dumps(dict(model=asdict(model), training=train)))


def class_weights(records: list[dict]) -> torch.Tensor:
    counts = torch.bincount(torch.tensor([r["label"] for r in records]), minlength=2)
    if (counts == 0).any():
        raise ValueError("Training needs both fake and live classes")
    return len(records) / (2 * counts.float())


def make_optimizer(model: nn.Module, settings: dict):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": settings["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=settings["learning_rate"], betas=(0.9, 0.999), eps=1e-8)


def lr_factor(epoch: int, settings: dict) -> float:
    warmup, epochs = settings["warmup_epochs"], settings["epochs"]
    if epoch < warmup:
        return (epoch + 1) / warmup
    floor = settings["min_learning_rate"] / settings["learning_rate"]
    # First post-warmup epoch starts at base LR, final epoch reaches the floor.
    progress = min(1.0, (epoch - warmup) / max(1, epochs - warmup - 1))
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, settings, device, *, training=False, epoch=0):
    generator = torch.Generator().manual_seed(settings["seed"] + epoch)
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=training,
                      num_workers=settings["num_workers"], pin_memory=device.type == "cuda",
                      worker_init_fn=seed_worker, generator=generator, persistent_workers=False)


def capture_rng() -> dict:
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                mps=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None)


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def environment(device: torch.device, precision: str) -> dict:
    info = dict(python=platform.python_version(), platform=platform.platform(),
                torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                pillow=PIL.__version__, sklearn=sklearn.__version__, numpy=np.__version__,
                cuda_runtime=torch.version.cuda, device=str(device), training_precision=precision)
    if device.type == "cuda":
        info.update(gpu=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)),
                    total_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
        # Driver reporting is best-effort; inference/training never depends on nvidia-smi.
        import subprocess
        try:
            info["nvidia_driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            info["nvidia_driver"] = "unavailable"
    return info


def export_best(output_dir: Path, checkpoint: dict) -> None:
    """best.pt is a recoverable inference export; last.pt owns the committed run state."""
    fields = ("format_version", "model_config", "config", "class_to_idx", "class_weights",
              "preprocessing", "augmentation", "environment", "dataset_fingerprint",
              "protocol", "cross_split_duplicate_groups", "best_metrics")
    payload = {key: checkpoint[key] for key in fields}
    payload.update(model=checkpoint["best_model"], epoch=checkpoint["best_epoch"])
    save_checkpoint(output_dir / "best.pt", payload)


def train_epoch(model, loader, optimizer, weights, device, settings, epoch, *, batch_progress=None):
    model.train()
    loss_sum, weight_sum, correct, examples = 0.0, 0.0, 0, 0
    start = time.perf_counter()
    use_amp = device.type == "cuda" and settings["precision"] == "bf16"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step, (images, labels, _) in enumerate(loader, 1):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext():
            logits = model(images)
        loss = F.cross_entropy(logits.float(), labels, weight=weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        batch_weight = weights[labels].sum().item()
        loss_sum += loss.item() * batch_weight
        weight_sum += batch_weight
        correct += (logits.argmax(-1) == labels).sum().item()
        examples += len(labels)
        if batch_progress is not None:
            peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
            batch_progress(step=step, loss=loss_sum / weight_sum, examples=examples,
                           accuracy=correct / examples, peak=peak)
    elapsed = time.perf_counter() - start
    return dict(train_loss=loss_sum / weight_sum, train_accuracy=correct / examples,
                train_seconds=elapsed, train_images_per_second=examples / elapsed)


def run_training(config: dict, data_root: Path, output_dir: Path, device_name="auto", resume: Path | None = None):
    console = Console()
    config = validate_config(config)
    settings = config["training"]
    data_root, output_dir = Path(data_root).resolve(), Path(output_dir).resolve()
    if output_dir.is_relative_to(data_root):
        raise ValueError("output_dir must be outside the source dataset")
    device = resolve_device(device_name)
    precision = "bf16" if device.type == "cuda" and settings["precision"] == "bf16" else "fp32"
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA BF16 was requested but this GPU does not support it")
    checkpoint = None
    if resume is not None:
        resume = Path(resume).resolve()
        if resume != output_dir / "last.pt":
            raise ValueError("Resume from this run's output_dir/last.pt; use a new run for changed settings")
        checkpoint = read_checkpoint(resume)
        if checkpoint["config"] != config or checkpoint["augmentation"] != AUGMENTATION:
            raise ValueError("Resume requires the same model, training and augmentation configuration")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty; use --resume or a new output directory")
    random.seed(settings["seed"])
    np.random.seed(settings["seed"] % 2**32)
    torch.manual_seed(settings["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    console.print("正在檢查資料集：逐張解碼並計算 SHA256，請稍候。")
    records, audit = audit_dataset(data_root)
    if checkpoint is not None and checkpoint["dataset_fingerprint"] != audit["fingerprint"]:
        raise ValueError("Dataset contents changed since checkpoint; refusing an inconsistent resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "data_manifest.csv", records)
    write_json(output_dir / "data_audit.json", audit)
    write_json(output_dir / "config.json", config)
    env = environment(device, precision)
    write_json(output_dir / "environment.json", env)
    console.print(f"裝置={device}，訓練精度={precision}；test 用於驗證與模型選擇。")
    console.print(f"保留跨訓練／驗證集重複內容群組：{audit['cross_split_duplicate_groups']}")

    spec = preprocessing_spec(config["model"]["image_size"][0])
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "test"]
    training = FASDataset(data_root, train_records, FASTransform(spec, training=True))
    validation = FASDataset(data_root, val_records, FASTransform(spec))
    val_loader = make_loader(validation, settings, device)
    weights = class_weights(train_records).to(device)
    model = LiFTClassifier(LiFTConfig(**config["model"])).to(device)
    optimizer = make_optimizer(model, settings)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_factor(epoch, settings))
    history, best, best_epoch, start_epoch, best_model = [], None, 0, 0, None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, best = checkpoint["history"], checkpoint["best_metrics"]
        best_epoch, start_epoch = checkpoint["best_epoch"], checkpoint["epoch"]
        best_model = checkpoint["best_model"]
        export_best(output_dir, checkpoint)
        restore_rng(checkpoint["rng"])
    for epoch in range(start_epoch, settings["epochs"]):
        loader = make_loader(training, settings, device, training=True, epoch=epoch)
        used_lr = optimizer.param_groups[0]["lr"]
        console.print(f"Epoch {epoch + 1}/{settings['epochs']} · LR={used_lr:.3g}", style="bold")
        with make_progress(console=console) as progress:
            train_progress = BatchProgress(progress, "Train", len(loader))
            val_progress = BatchProgress(progress, "Val  ", len(val_loader), validation=True)
            stats = train_epoch(model, loader, optimizer, weights, device, settings, epoch + 1,
                                batch_progress=train_progress.start())
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            metrics, _ = evaluate(model, val_loader, device, batch_progress=val_progress.start())
        improved = is_better(metrics, best)
        if improved:
            best, best_epoch = metrics, epoch + 1
            best_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(dict(epoch=epoch + 1, learning_rate=used_lr, **stats,
                            **{"val_" + key: value for key, value in metrics.items()}))
        scheduler.step()
        payload = dict(format_version=1, model=model.state_dict(), model_config=config["model"],
                       config=config, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       epoch=epoch + 1, best_metrics=best, best_epoch=best_epoch, best_model=best_model, history=history,
                       rng=capture_rng(), class_to_idx=CLASSES, class_weights=weights.cpu(),
                       preprocessing=spec, augmentation=AUGMENTATION, environment=env,
                       dataset_fingerprint=audit["fingerprint"], protocol=audit["protocol"],
                       cross_split_duplicate_groups=audit["cross_split_duplicate_groups"])
        save_checkpoint(output_dir / "last.pt", payload)
        if improved:
            export_best(output_dir, payload)
        write_csv(output_dir / "history.csv", history)
        console.print(
            f"Epoch {epoch + 1}: loss={stats['train_loss']:.5f}, train acc={stats['train_accuracy']:.2%}, "
            f"val acc={metrics['accuracy']:.2%}, val ACER={metrics['acer']:.4%}, "
            f"best={best['acer']:.4%}\n"
            f"val loss={metrics['loss']:.5f}, APCER={metrics['apcer']:.4%}, "
            f"BPCER={metrics['bpcer']:.4%}, AUC={metrics['roc_auc']:.6f}, "
            f"最佳 epoch={best_epoch}" + ("（已更新 best.pt）" if improved else ""),
            markup=False,
        )
    # History in last.pt is authoritative if interruption occurred during CSV writing.
    write_csv(output_dir / "history.csv", history)
    model, selected = load_model(output_dir / "best.pt", device)
    results = {}
    for loops in sorted({1, 2, 4, 8, config["model"]["num_loops"]}):
        with make_progress(console=console) as progress:
            val_progress = BatchProgress(progress, f"Val K={loops}", len(val_loader), validation=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            metrics, predictions = evaluate(model, val_loader, device, num_loops=loops,
                                             batch_progress=val_progress.start())
        results[str(loops)] = metrics
        write_csv(output_dir / f"validation_predictions_k{loops}.csv", predictions)
    report = dict(best_epoch=selected["epoch"], default_num_loops=config["model"]["num_loops"],
                  evaluation_precision="fp32", protocol=audit["protocol"],
                  cross_split_duplicate_groups=audit["cross_split_duplicate_groups"],
                  limitation=audit["limitation"], metrics_by_num_loops=results)
    write_json(output_dir / "validation_report.json", report)
    return report
