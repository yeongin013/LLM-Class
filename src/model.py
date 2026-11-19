"""B0 fine-tuning (PLAN.md stage "기준 모델 B0"): bert-base-uncased and the DistilBERT comparison model.

Run: python src/model.py [--models bert distilbert] [--seeds 13 42 2026]          B0 training
     python src/model.py --stage exit [--models ...] [--seeds ...]                   T2 probing + early exit
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


# ---------------------------------------------------------------- T2: probing + early exit

TAUS = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999, 1.01]  # 1.01 = never exit early


def backbone(model):
    base = getattr(model, "bert", None) or model.distilbert
    layers = base.encoder.layer if hasattr(base, "encoder") else base.transformer.layer
    return base, layers


def final_head(model, h):
    """The fine-tuned model's own head on last-layer hidden states (BERT: pooler + classifier)."""
    if hasattr(model, "bert"):
        return model.classifier(model.bert.pooler(h))
    return model.classifier(torch.relu(model.pre_classifier(h[:, 0])))


@torch.inference_mode()
def layer_cls(model, tok, texts):
    """[CLS] vector after every layer, shape (L, N, d), and the final-head attack probability."""
    model.eval()
    feats, probs = [], []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], padding="longest", return_tensors="pt").to(DEVICE)
        out = model(**enc, output_hidden_states=True)
        feats.append(torch.stack([h[:, 0] for h in out.hidden_states[1:]]).float().cpu())
        probs += torch.softmax(out.logits.float(), -1)[:, 1].tolist()
    return torch.cat(feats, 1), probs


class EarlyExit:
    """Batch-1 inference that stops at the first layer whose exit head is at least tau confident."""

    def __init__(self, model, heads, tau, device):
        self.model, self.tau, self.device = model.to(device).eval(), tau, device
        self.heads = [(w.to(device), b.to(device)) for w, b in heads]

    @torch.inference_mode()
    def __call__(self, enc):
        base, layers = backbone(self.model)
        h = base.embeddings(input_ids=enc["input_ids"].to(self.device))
        for depth, layer in enumerate(layers, 1):
            h = layer(h)
            if depth < len(layers):
                w, b = self.heads[depth - 1]
                p = torch.sigmoid(h[0, 0] @ w + b).item()  # .item() syncs the device: part of the real cost
                if max(p, 1 - p) >= self.tau:
                    return p, depth
        return torch.softmax(final_head(self.model, h).float(), -1)[0, 1].item(), len(layers)


def simulate_exit(P, tau):
    """P: (L, N) attack prob per layer (last row = final head). Returns exit prob and depth per sample."""
    conf = torch.maximum(P, 1 - P) >= tau
    conf[-1] = True
    depth = conf.int().argmax(0)  # first confident layer (0-based)
    return P[depth, torch.arange(P.shape[1])].tolist(), (depth + 1).tolist()


def exit_flops(n, depth, L, d):
    from bench import theory_flops
    if depth == L:
        return theory_flops(n, L, d)
    return 2 * (depth * (12 * n * d * d + 2 * n * n * d) + d)  # linear exit head: d multiply-adds


def early_exit(keys, seeds, latency_seed=13, n_latency=500, f1_drop=0.005):
    import statistics
    from sklearn.linear_model import LogisticRegression
    from bench import config_dims, latency

    Xtr, ytr = load("train")
    Xva, yva = load("validation")
    out = {"probe": {}, "exit": {}, "latency": {}, "rule": f"smallest mean depth with val macro F1 >= full - {f1_drop}"}
    for key in keys:
        for seed in seeds:
            run = f"{key}_s{seed}"
            path = ROOT / "checkpoints" / run
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
            L, d = config_dims(model.config)
            Ftr, _ = layer_cls(model, tok, Xtr)
            Fva, full_probs = layer_cls(model, tok, Xva)

            heads, P, probe = [], [], []
            for layer in range(L):
                clf = LogisticRegression(max_iter=2000).fit(Ftr[layer].numpy(), ytr)
                p = clf.predict_proba(Fva[layer].numpy())[:, 1]
                probe.append(evaluate(yva, p) | {"layer": layer + 1})
                heads.append((torch.tensor(clf.coef_[0], dtype=torch.float32),
                              torch.tensor(clf.intercept_[0], dtype=torch.float32)))
                P.append(torch.tensor(p, dtype=torch.float32))
            P[-1] = torch.tensor(full_probs)  # deepest exit = the model's own head, so "never exit" reproduces B0
            P = torch.stack(P)
            torch.save(heads[:-1], path / "exit_heads.pt")
            out["probe"][run] = probe

            lens = [len(ids) for ids in tok(Xva)["input_ids"]]
            full_flops = sum(exit_flops(n, L, L, d) for n in lens)
            sweep = []
            for tau in TAUS:
                probs, depths = simulate_exit(P, tau)
                sweep.append(evaluate(yva, probs) | {
                    "tau": tau, "mean_depth": statistics.mean(depths),
                    "flops_vs_full_actual_len": sum(exit_flops(n, k, L, d) for n, k in zip(lens, depths)) / full_flops})
            full = sweep[-1]["macro_f1"]
            chosen = min((r for r in sweep if r["macro_f1"] >= full - f1_drop), key=lambda r: r["mean_depth"])
            out["exit"][run] = {"layers": L, "sweep": sweep, "chosen_tau": chosen["tau"]}
            print(f"{run}: probe F1 by layer {[round(r['macro_f1'], 3) for r in probe]}; chosen tau {chosen['tau']} "
                  f"depth {chosen['mean_depth']:.2f}/{L} F1 {chosen['macro_f1']:.4f} (full {full:.4f})")

            if seed != latency_seed:
                continue
            sample = [tok(t, return_tensors="pt") for t in Xva[:n_latency]]
            never = EarlyExit(model, heads[:-1], 1.01, DEVICE)
            gap = max(abs(never(enc)[0] - p) for enc, p in zip(sample[:100], full_probs))
            assert gap < 1e-4, f"layer-by-layer loop does not reproduce B0 (gap {gap})"
            out["exit"][run]["loop_vs_b0_max_prob_gap"] = gap
            lat = {}
            for device in ["mps", "cpu"]:
                threads = torch.get_num_threads()
                if device == "cpu":
                    torch.set_num_threads(1)
                model.to(device)

                def hf(enc, device=device):
                    with torch.inference_mode():
                        model(**{k: v.to(device) for k, v in enc.items()})

                res = {"hf_full": latency(hf, sample, device)}
                for tau in sorted({chosen["tau"], 1.01}):
                    res[f"exit_tau_{tau}"] = latency(EarlyExit(model, heads[:-1], tau, device), sample, device)
                lat[device if device != "cpu" else "cpu_1thread"] = res
                torch.set_num_threads(threads)
            model.to(DEVICE)
            out["latency"][run] = lat
            print(run, {dv: {k: round(v["median_ms"], 2) for k, v in r.items()} for dv, r in lat.items()})

    (RESULTS / "t2_exit.json").write_text(json.dumps(out, indent=2))
    plot_exit(out)


def plot_exit(out):
    import statistics
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"bert": "#2a78d6", "distilbert": "#eb6834"}  # categorical slots 1-2, validated light mode
    ink, muted, surface = "#1f1f1e", "#6b6a64", "#fcfcfb"
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=surface)
    for key, color in colors.items():
        runs = [v for k, v in out["probe"].items() if k.startswith(key + "_")]
        if not runs:
            continue
        layers = [r["layer"] for r in runs[0]]
        f1 = [[run[i]["macro_f1"] for run in runs] for i in range(len(layers))]
        left.fill_between(layers, [min(v) for v in f1], [max(v) for v in f1], color=color, alpha=0.15, lw=0)
        left.plot(layers, [statistics.mean(v) for v in f1], color=color, lw=2, marker="o", ms=4)
        left.annotate(key, (layers[-1], statistics.mean(f1[-1])), xytext=(6, -3), textcoords="offset points",
                      color=ink, fontsize=9)

        sweeps = [v["sweep"] for k, v in out["exit"].items() if k.startswith(key + "_")]
        depth = [statistics.mean(s[i]["mean_depth"] for s in sweeps) for i in range(len(sweeps[0]))]
        f1x = [statistics.mean(s[i]["macro_f1"] for s in sweeps) for i in range(len(sweeps[0]))]
        right.plot(depth, f1x, color=color, lw=2, marker="o", ms=4)
        right.annotate(key, (depth[-1], f1x[-1]), xytext=(6, -3), textcoords="offset points", color=ink, fontsize=9)
    left.set_title("Probe: [CLS] at each layer (mean, band = seed min-max)", color=ink, fontsize=10, loc="left")
    left.set_xlabel("layer", color=ink)
    left.set_ylabel("validation macro F1", color=ink)
    right.set_title("Early exit: tau sweep (mean of 3 seeds)", color=ink, fontsize=10, loc="left")
    right.set_xlabel("mean layers run per sentence", color=ink)
    right.set_ylabel("validation macro F1", color=ink)
    for ax in (left, right):
        ax.set_facecolor(surface)
        ax.grid(color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=muted)
    fig.tight_layout()
    fig.savefig(RESULTS / "t2_exit.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["b0", "exit"], default="b0")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2026])
    args = ap.parse_args()
    if args.stage == "exit":
        early_exit(args.models, args.seeds)
    else:
        for key in args.models:
            for seed in args.seeds:
                train(key, seed)
        summarize()
