"""B0 fine-tuning (PLAN.md stage "기준 모델 B0"): bert-base-uncased and the DistilBERT comparison model.

Run: python src/model.py [--models bert distilbert] [--seeds 13 42 2026]
Per run writes checkpoints/<model>_s<seed>/ (best epoch by validation macro F1; gitignored)
and results/b0/<model>_s<seed>.json (per-epoch validation metrics, timing) plus validation probabilities.
test_clean is never touched here; it is scored once in the final stage.
"""
import argparse
import csv
import json
import random
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bench import CLEAN, MODELS, RESULTS, ROOT, evaluate

LR, BATCH, EPOCHS, WARMUP, WEIGHT_DECAY = 2e-5, 32, 3, 0.1, 0.01
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load(split):
    with open(CLEAN / f"{split}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [int(r["label"]) for r in rows]


def loader(tok, texts, labels, shuffle, seed=0):
    enc = tok(texts)["input_ids"]
    items = [{"input_ids": ids, "labels": y} for ids, y in zip(enc, labels)]

    def collate(batch):
        # Dynamic padding: pad to the longest sentence in the batch (masked, so results match fixed padding).
        return tok.pad(batch, return_tensors="pt")

    g = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=BATCH, shuffle=shuffle, collate_fn=collate, generator=g)


@torch.inference_mode()
def predict(model, data):
    model.eval()
    probs = []
    for batch in data:
        batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
        probs += torch.softmax(model(**batch).logits.float(), -1)[:, 1].tolist()
    return probs


def train(key, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    name = MODELS[key]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(DEVICE)

    Xtr, ytr = load("train")
    Xva, yva = load("validation")
    train_dl = loader(tok, Xtr, ytr, shuffle=True, seed=seed)
    val_dl = loader(tok, Xva, yva, shuffle=False)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=LR)
    total = EPOCHS * len(train_dl)
    warm = int(WARMUP * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    out_dir = ROOT / "checkpoints" / f"{key}_s{seed}"
    history, best = [], None
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        te = time.perf_counter()
        for batch in train_dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            model(**batch).loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        probs = predict(model, val_dl)
        m = evaluate(yva, probs) | {"epoch": epoch, "train_sec": time.perf_counter() - te}
        history.append(m)
        print(f"{key} s{seed} epoch {epoch}: val macro F1 {m['macro_f1']:.4f}  AUC {m['roc_auc']:.4f}  ({m['train_sec']:.0f}s)")
        if best is None or m["macro_f1"] > best["macro_f1"]:
            best, best_probs = m, probs
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)

    run = {"model": name, "seed": seed, "device": DEVICE, "hparams": {"lr": LR, "batch": BATCH, "epochs": EPOCHS,
           "warmup": WARMUP, "weight_decay": WEIGHT_DECAY}, "history": history, "best": best,
           "total_sec": time.perf_counter() - t0}
    out = RESULTS / "b0"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{key}_s{seed}.json").write_text(json.dumps(run, indent=2))
    with open(out / f"{key}_s{seed}_val_probs.csv", "w", newline="") as f:
        csv.writer(f).writerows([["row", "label", "prob_attack"]] + [[i, y, f"{p:.6f}"] for i, (y, p) in enumerate(zip(yva, best_probs))])
    return run


def summarize():
    import statistics
    runs = [json.loads(p.read_text()) for p in sorted((RESULTS / "b0").glob("*_s*.json"))]
    summary = {}
    for key, name in MODELS.items():
        rs = [r for r in runs if r["model"] == name]
        if not rs:
            continue
        summary[key] = {"seeds": [r["seed"] for r in rs], "best_epochs": [r["best"]["epoch"] for r in rs]}
        for metric in ["macro_f1", "attack_recall", "benign_fpr", "roc_auc"]:
            v = [r["best"][metric] for r in rs]
            summary[key][metric] = {"mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0.0}
        summary[key]["train_min_per_run"] = statistics.mean(r["total_sec"] for r in rs) / 60
    (RESULTS / "b0" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2026])
    args = ap.parse_args()
    for key in args.models:
        for seed in args.seeds:
            train(key, seed)
    summarize()
