"""Fine-tune wav2vec 2.0 (or WavLM / HuBERT) to recognise the phrases, or evaluate it untrained. One script for every run.

Each run is a set of CLI options, not a new script:

| Run (task) | Options | Results |
|---|---|---|
| 15 epochs + white noise (T15) | ``--epochs 15 --noise white`` | ``results/w2v2_15ep_noise/`` |
| 8 epochs + white noise (T16) | ``--epochs 8 --noise white`` | ``results/w2v2_8ep_noise/`` |
| 3 epochs (T17) | ``--epochs 3`` | ``results/w2v2_3ep/`` |
| Untrained baseline (T18) | ``--untrained`` | ``results/w2v2_untrained/`` |
| 15 epochs, no noise (T19) | ``--epochs 15`` | ``results/w2v2_15ep/`` |
| WavLM Base+ (T44) | ``--model microsoft/wavlm-base-plus --epochs 15 --noise white`` | ``results/wavlm_15ep_noise/`` |
| HuBERT Base (T45) | ``--model facebook/hubert-base-ls960 --epochs 15 --noise white`` | ``results/hubert_15ep_noise/`` |

For every split: load the processed clips (padded to 6 s), fine-tune with AdamW and a
linear warm-up schedule (settings: ``wav2vec2`` section of the config), evaluate on the
**validation** part after every epoch, keep the best epoch, and score it once on the
test part. White noise (``--noise white``) is added to training clips only. The test
predictions go to ``support.evaluate.save_run``; the best model of each split is saved
to ``outputs/models/<run>/<dataset>/<split>/`` (gitignored; copy it to the team Drive).

Usage::

    python -m support.train_wav2vec2 --dataset fsc --epochs 15 --noise white            # random + 5 folds
    python -m support.train_wav2vec2 --dataset l2arctic --epochs 3 --split fold2
    python -m support.train_wav2vec2 --dataset fsc --untrained
    python -m support.train_wav2vec2 --dataset fsc --epochs 1 --split random --limit 40 # quick check
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from support.audio import pad_to_duration
from support.config import load_config, resolve_path
from support.dataset import load_labels, load_split, parse_splits, read_processed
from support.device import pick_device
from support.evaluate import save_run
from support.noise import augment


def model_tag(model_name: str | None) -> str:
    """Short family name of a checkpoint: ``w2v2``, ``wavlm``, ``hubert`` or the checkpoint's own name."""
    name = (model_name or "wav2vec2").lower()
    for key, tag in (("wavlm", "wavlm"), ("hubert", "hubert"), ("wav2vec2", "w2v2")):
        if key in name:
            return tag
    return Path(name).name.replace("-", "_")


def default_run_name(epochs: int, noise: str, untrained: bool, model_name: str | None = None) -> str:
    """``w2v2_untrained``, ``w2v2_3ep``, ``w2v2_15ep_noise``, ``wavlm_15ep_noise``, ``hubert_15ep_noise`` …"""
    tag = model_tag(model_name)
    if untrained:
        return f"{tag}_untrained"
    return f"{tag}_{epochs}ep" + ("_noise" if noise == "white" else "")


class ClipDataset(Dataset):
    """Processed clips padded to the fixed input length, with optional noise augmentation.

    Augmentation must only ever be switched on for the training part.
    """

    def __init__(self, frame: pd.DataFrame, label_index: dict[str, int], sample_rate: int, max_duration_s: float,
                 augment_cfg: dict[str, Any] | None = None, seed: int = 42):
        self.paths = list(frame["path"])
        self.labels = [label_index[label] for label in frame["label"]]
        self.sample_rate, self.max_duration_s = sample_rate, max_duration_s
        self.augment_cfg = augment_cfg
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        samples = read_processed(self.paths[i])
        if self.augment_cfg is not None:
            samples, _ = augment(samples, self.rng, float(self.augment_cfg["p"]),
                                 tuple(float(v) for v in self.augment_cfg["snr_db"]))
        return pad_to_duration(samples, self.sample_rate, self.max_duration_s), self.labels[i]


def make_collate(feature_extractor, sample_rate: int):
    """Batch clips with the model's own feature extractor (zero mean, unit variance)."""
    def collate(batch):
        audio, labels = zip(*batch)
        inputs = feature_extractor(list(audio), sampling_rate=sample_rate, return_tensors="pt")
        return inputs["input_values"], torch.tensor(labels, dtype=torch.long)
    return collate


def build_model(model_name: str, labels: Sequence[str], freeze_feature_encoder: bool = True):
    """wav2vec 2.0 with a new classification head for ``labels``."""
    from transformers import AutoModelForAudioClassification

    model = AutoModelForAudioClassification.from_pretrained(
        model_name, num_labels=len(labels), label2id={l: i for i, l in enumerate(labels)},
        id2label={i: l for i, l in enumerate(labels)}, ignore_mismatched_sizes=True)
    if freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    return model


@torch.no_grad()
def predict(model, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Predicted class index, its softmax probability, and the mean loss over a loader."""
    model.eval()
    preds, confs, losses, n = [], [], 0.0, 0
    for inputs, labels in loader:
        out = model(input_values=inputs.to(device), labels=labels.to(device))
        probs = torch.softmax(out.logits.float(), dim=-1)
        conf, idx = probs.max(dim=-1)
        preds.append(idx.cpu().numpy())
        confs.append(conf.cpu().numpy())
        losses += out.loss.item() * len(labels)
        n += len(labels)
    return np.concatenate(preds), np.concatenate(confs), losses / max(n, 1)


def train_split(dataset: str, split: str, cfg: dict[str, Any], epochs: int, noise: str, untrained: bool,
                model_name: str, device: str, save_dir: Path | None = None,
                limit: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fine-tune (or only evaluate) on one split and predict its test clips.

    Returns:
        Test predictions (``utt_id, speaker_id, label, pred, confidence``) and run details
        (best epoch, learning curve, sizes).
    """
    wcfg, acfg = cfg["wav2vec2"], cfg["audio"]
    seed = int(cfg.get("seed", 42))
    sr, max_s = int(acfg["sample_rate"]), float(acfg["max_duration_s"])
    labels = list(load_labels(dataset, cfg)["phrase_id"])
    index = {l: i for i, l in enumerate(labels)}
    parts = load_split(dataset, split, cfg)
    if limit:
        parts = {p: df.head(limit) for p, df in parts.items()}

    from transformers import AutoFeatureExtractor, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = build_model(model_name, labels, bool(wcfg.get("freeze_feature_encoder", True))).to(device)
    collate = make_collate(extractor, sr)
    batch = int(wcfg["batch_size"])
    loaders = {p: DataLoader(ClipDataset(parts[p], index, sr, max_s), batch_size=batch, shuffle=False,
                             collate_fn=collate) for p in ("val", "test")}
    val_true = np.array([index[l] for l in parts["val"]["label"]])

    curve: list[dict[str, Any]] = []
    best_epoch, best_acc = 0, -1.0
    if untrained:
        pred, _, loss = predict(model, loaders["val"], device)
        best_acc = float(np.mean(pred == val_true))
        curve.append({"epoch": 0, "train_loss": None, "val_loss": round(loss, 4), "val_accuracy": best_acc})
    else:
        augment_cfg = cfg["augmentation"] if noise == "white" else None
        train_set = ClipDataset(parts["train"], index, sr, max_s, augment_cfg, seed)
        train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=collate,
                                  generator=torch.Generator().manual_seed(seed))
        accum = int(wcfg["grad_accum"])
        steps = math.ceil(len(train_loader) / accum) * epochs
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=float(wcfg["learning_rate"]), weight_decay=float(wcfg["weight_decay"]))
        scheduler = get_linear_schedule_with_warmup(optimizer, int(float(wcfg["warmup_ratio"]) * steps), steps)
        use_fp16 = device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
        best_state = None
        for epoch in range(1, epochs + 1):
            model.train()
            total, seen = 0.0, 0
            optimizer.zero_grad()
            for step, (inputs, y) in enumerate(train_loader, start=1):
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    loss = model(input_values=inputs.to(device), labels=y.to(device)).loss
                scaler.scale(loss / accum).backward()
                total += loss.detach().item() * len(y)
                seen += len(y)
                if step % accum == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            pred, _, val_loss = predict(model, loaders["val"], device)
            acc = float(np.mean(pred == val_true))
            curve.append({"epoch": epoch, "train_loss": round(total / seen, 4), "val_loss": round(val_loss, 4),
                          "val_accuracy": acc})
            print(f"{split} · epoch {epoch}/{epochs}: train loss {total / seen:.4f}, validation accuracy {acc:.3f}")
            if acc > best_acc:  # ties keep the earlier epoch
                best_epoch, best_acc = epoch, acc
                best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)

    pred, conf, _ = predict(model, loaders["test"], device)
    test = parts["test"]
    predictions = pd.DataFrame({"utt_id": test["utt_id"], "speaker_id": test["speaker_id"], "label": test["label"],
                                "pred": [labels[i] for i in pred], "confidence": np.round(conf, 4)})
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        extractor.save_pretrained(save_dir)
    info = {"trained": not untrained, "best_epoch": best_epoch, "val_accuracy": best_acc, "curve": curve,
            "n_train": len(parts["train"]), "n_val": len(parts["val"]), "n_test": len(test)}
    return predictions, info


def run(dataset: str, splits: Sequence[str], epochs: int = 15, noise: str = "none", untrained: bool = False,
        run_name: str | None = None, cfg: dict[str, Any] | None = None, model_name: str | None = None,
        device: str | None = None, save_models: bool = True, limit: int | None = None) -> dict[str, Any]:
    """Train and evaluate on several splits and save the results with ``support.evaluate``."""
    cfg = cfg if cfg is not None else load_config()
    if noise not in {"none", "white"}:
        raise ValueError("noise must be 'none' or 'white'")
    model_name = model_name or cfg["wav2vec2"]["model"]
    run_name = run_name or default_run_name(epochs, noise, untrained, model_name)
    device = pick_device(device)
    models_root = resolve_path(cfg, "outputs") / "models" / run_name / dataset
    preds, details = {}, {}
    for split in splits:
        save_dir = models_root / split if save_models else None
        preds[split], details[split] = train_split(dataset, split, cfg, epochs, noise, untrained, model_name,
                                                   device, save_dir, limit)
        print(f"{run_name} · {dataset} · {split}: best epoch {details[split]['best_epoch']}, "
              f"validation accuracy {details[split]['val_accuracy']:.3f}")
    labels = list(load_labels(dataset, cfg)["phrase_id"])
    extra = {"model": model_name, "epochs": 0 if untrained else epochs, "noise": noise, "untrained": untrained,
             "device": device, "settings": cfg["wav2vec2"],
             "augmentation": cfg["augmentation"] if noise == "white" and not untrained else None,
             "selection": "best validation accuracy over epochs", "limit": limit, "splits": details}
    return save_run(run_name, dataset, preds, labels, extra, cfg)


def main() -> None:
    """Run one wav2vec 2.0 configuration on one dataset."""
    parser = argparse.ArgumentParser(description="Fine-tune or evaluate wav2vec 2.0 on a TenVoices subset.")
    parser.add_argument("--dataset", required=True, help="fsc or l2arctic")
    parser.add_argument("--split", default="all", type=parse_splits, help="all (default), folds, random or foldN")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--noise", default="none", choices=["none", "white"], help="white-noise augmentation")
    parser.add_argument("--untrained", action="store_true", help="evaluate with an untrained head (baseline)")
    parser.add_argument("--run", default=None, help="results folder name (default from the options)")
    parser.add_argument("--model", default=None, help="checkpoint (default: config wav2vec2.model)")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N clips of each part (quick check)")
    parser.add_argument("--no-save-models", action="store_true", help="do not keep the trained models")
    parser.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    args = parser.parse_args()
    summary = run(args.dataset, args.split, args.epochs, args.noise, args.untrained, args.run,
                  load_config(args.config), args.model, args.device, not args.no_save_models, args.limit)
    print(json.dumps({k: summary[k] for k in ("random", "folds") if k in summary}, indent=2)[:1500])


if __name__ == "__main__":
    main()
