"""B0 fine-tuning (PLAN.md stage "기준 모델 B0"): bert-base-uncased and the DistilBERT comparison model.

Run: python src/model.py [--models bert distilbert] [--seeds 13 42 2026]          B0 training
     python src/model.py --stage exit [--models ...] [--seeds ...] [--name ...]      T2 probing + early exit
       (GQA: --models bert_gqa1 --name t2_exit_gqa1)
     python src/model.py --stage prune [--models ...] [--seeds ...]                  T3 token pruning
     python src/model.py --stage gqa [--seeds ...]                                   T4 GQA conversion (BERT)
Per run writes checkpoints/<model>_s<seed>/ (best epoch by validation macro F1; gitignored)
and results/b0/<model>_s<seed>.json (per-epoch validation metrics, timing) plus validation probabilities.
test_clean is never touched here; it is scored once in the final stage.
"""
import argparse
import csv
import json
import math
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


def train(key, seed, model=None, epochs=EPOCHS, tag=None, out_sub="b0"):
    """Fine-tune `key` from the hub, or continue training a given `model` (T4 uptraining)."""
    random.seed(seed)
    torch.manual_seed(seed)
    name = MODELS[key]
    tag = tag or f"{key}_s{seed}"
    tok = AutoTokenizer.from_pretrained(name)
    model = (model or AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)).to(DEVICE)

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
    total = epochs * len(train_dl)
    warm = int(WARMUP * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    out_dir = ROOT / "checkpoints" / tag
    history, best = [], None
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
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
        print(f"{tag} epoch {epoch}: val macro F1 {m['macro_f1']:.4f}  AUC {m['roc_auc']:.4f}  ({m['train_sec']:.0f}s)")
        if best is None or m["macro_f1"] > best["macro_f1"]:
            best, best_probs = m, probs
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)

    run = {"model": name, "seed": seed, "device": DEVICE, "hparams": {"lr": LR, "batch": BATCH, "epochs": epochs,
           "warmup": WARMUP, "weight_decay": WEIGHT_DECAY}, "history": history, "best": best,
           "total_sec": time.perf_counter() - t0}
    out = RESULTS / out_sub
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(run, indent=2))
    with open(out / f"{tag}_val_probs.csv", "w", newline="") as f:
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


def early_exit(keys, seeds, latency_seed=13, n_latency=500, f1_drop=0.005, name="t2_exit"):
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
            model = load_model(path).to(DEVICE)
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

    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=2))
    plot_exit(out, name)


def plot_exit(out, name="t2_exit"):
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
    fig.savefig(RESULTS / f"{name}.png", dpi=150)


# ---------------------------------------------------------------- T3: attention-score token pruning

PRUNE_AT = {"bert": [4, 6, 8], "distilbert": [2, 3, 4]}  # prune after this layer
KEEP = [0.5, 0.7]


def cls_attention(model, layer, h):
    """Head-averaged attention from [CLS] to every token, as computed inside `layer` on its input h (1, n, d)."""
    if hasattr(model, "bert"):
        q_lin, k_lin, heads = layer.attention.self.query, layer.attention.self.key, model.config.num_attention_heads
    else:
        q_lin, k_lin, heads = layer.attention.q_lin, layer.attention.k_lin, model.config.n_heads
    n, d = h.shape[1], h.shape[2]
    q = q_lin(h[:, :1]).view(heads, 1, d // heads)
    k = k_lin(h).view(n, heads, d // heads).transpose(0, 1)
    return torch.softmax(q @ k.transpose(1, 2) / (d // heads) ** 0.5, -1).mean(0)[0]  # (n,)


@torch.inference_mode()
def pruned_forward(model, input_ids, prune_at, keep):
    """Batch-1 forward that keeps the top `keep` fraction of tokens after layer `prune_at`, by [CLS] attention.
    [CLS] and the final [SEP] are always kept; token order is preserved."""
    base, layers = backbone(model)
    h = base.embeddings(input_ids=input_ids)
    for depth, layer in enumerate(layers, 1):
        if depth == prune_at + 1:
            n = h.shape[1]
            m = max(2, math.ceil(keep * n))  # at least [CLS] + [SEP]
            if m < n:
                score = cls_attention(model, layer, h)
                score[0] = score[-1] = float("inf")
                idx = score.topk(m).indices.sort().values
                h = h[:, idx]
        h = layer(h)
    return torch.softmax(final_head(model, h).float(), -1)[0, 1].item()


def prune_flops(n, prune_at, keep, L, d):
    from bench import theory_flops
    m = max(2, math.ceil(keep * n)) if prune_at < L else n
    per_layer = lambda t: 12 * t * d * d + 2 * t * t * d
    return theory_flops(n, L, d) - 2 * (L - prune_at) * (per_layer(n) - per_layer(m))


def token_pruning(keys, seeds, latency_seed=13, n_latency=500):
    import statistics
    from bench import config_dims, latency

    Xva, yva = load("validation")
    out = {"accuracy": {}, "latency": {}}
    for key in keys:
        for seed in seeds:
            run = f"{key}_s{seed}"
            path = ROOT / "checkpoints" / run
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval()
            L, d = config_dims(model.config)
            encs = [tok(t, return_tensors="pt")["input_ids"].to(DEVICE) for t in Xva]
            lens = [e.shape[1] for e in encs]
            full = [pruned_forward(model, e, L, 1.0) for e in encs]  # prune_at = L: no pruning
            res = {"full": evaluate(yva, full) | {"flops_vs_full": 1.0}}
            for k in PRUNE_AT[key]:
                for r in KEEP:
                    probs = [pruned_forward(model, e, k, r) for e in encs]
                    res[f"after{k}_keep{r}"] = evaluate(yva, probs) | {
                        "flops_vs_full": sum(prune_flops(n, k, r, L, d) for n in lens)
                        / sum(prune_flops(n, L, 1.0, L, d) for n in lens),
                        "pred_changed_vs_full": sum((a >= 0.5) != (b >= 0.5) for a, b in zip(full, probs))}
            out["accuracy"][run] = res
            print(run, {c: (round(v["macro_f1"], 4), round(v["flops_vs_full"], 3)) for c, v in res.items()})

            if seed != latency_seed:
                continue
            sample = [tok(t, return_tensors="pt")["input_ids"] for t in Xva[:n_latency]]
            lat = {}
            for device in ["mps", "cpu"]:
                threads = torch.get_num_threads()
                if device == "cpu":
                    torch.set_num_threads(1)
                model.to(device)
                res = {}
                for k, r in [(L, 1.0)] + [(k, r) for k in PRUNE_AT[key] for r in KEEP]:
                    name = "full" if k == L else f"after{k}_keep{r}"
                    res[name] = latency(lambda e, k=k, r=r: pruned_forward(model, e.to(device), k, r), sample, device)
                lat[device if device != "cpu" else "cpu_1thread"] = res
                torch.set_num_threads(threads)
            model.to(DEVICE)
            out["latency"][run] = lat
            print(run, {dv: {c: round(v["mean_ms"], 2) for c, v in r.items()} for dv, r in lat.items()})

    summary = {}
    for key in keys:
        runs = [v for k, v in out["accuracy"].items() if k.startswith(key + "_")]
        summary[key] = {c: {m: statistics.mean(r[c][m] for r in runs) for m in ["macro_f1", "attack_recall", "benign_fpr",
                                                                              "flops_vs_full"]} for c in runs[0]}
    out["summary_mean_over_seeds"] = summary
    (RESULTS / "t3_prune.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------- T4: MHA -> GQA conversion + uptraining

GROUPS = [6, 4, 2, 1]  # K/V heads after conversion (BERT has 12)
UPTRAIN_EPOCHS = 1


def to_gqa(model, g, mean_pool=True):
    """Shrink every self-attention's key/value projections to g heads (mean of each group of 12/g heads).
    HF's SDPA attention repeats K/V per group when the module has `num_key_value_groups`."""
    H, d = model.config.num_attention_heads, model.config.hidden_size
    dh, r = d // H, H // g
    for layer in model.bert.encoder.layer:
        att = layer.attention.self
        for name in ("key", "value"):
            old = getattr(att, name)
            new = torch.nn.Linear(d, g * dh).to(old.weight.device)
            if mean_pool:
                with torch.no_grad():
                    new.weight.copy_(old.weight.view(g, r, dh, d).mean(1).reshape(g * dh, d))
                    new.bias.copy_(old.bias.view(g, r, dh).mean(1).reshape(g * dh))
            setattr(att, name, new)
        att.num_key_value_groups = r
    model.config.gqa_groups = g
    return model


def load_model(path):
    """from_pretrained, rebuilding the GQA shapes first when the checkpoint is a converted model."""
    from safetensors.torch import load_file
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(path)
    g = getattr(cfg, "gqa_groups", None)
    if not g:
        return AutoModelForSequenceClassification.from_pretrained(path)
    # from_pretrained handles the saved-name mapping (LayerNorm gamma/beta) but not the shrunk K/V shapes,
    # so load everything else through it and copy the K/V tensors in afterwards.
    model = to_gqa(AutoModelForSequenceClassification.from_pretrained(path, ignore_mismatched_sizes=True), g,
                   mean_pool=False)
    kv = {k: v for k, v in load_file(path / "model.safetensors").items() if ".attention.self.key." in k
          or ".attention.self.value." in k}
    assert len(kv) == 4 * cfg.num_hidden_layers
    missing, unexpected = model.load_state_dict(kv, strict=False)
    assert not unexpected
    return model


def gqa_flops(n, L, d, g, H=12):
    from bench import theory_flops
    return theory_flops(n, L, d) - 2 * L * 2 * n * d * d * (1 - g / H)  # K and V projections shrink to g/H


def gqa(seeds, latency_seed=13, n_latency=500):
    import statistics
    from bench import config_dims, latency, measured_flops

    Xva, yva = load("validation")
    out = {"uptrain_epochs": UPTRAIN_EPOCHS, "runs": {}, "latency": {}}
    for seed in seeds:
        b0 = ROOT / "checkpoints" / f"bert_s{seed}"
        tok = AutoTokenizer.from_pretrained(b0)
        val_dl = loader(tok, Xva, yva, shuffle=False)
        ref = predict(load_model(b0).to(DEVICE), val_dl)
        same = predict(to_gqa(load_model(b0), 12).to(DEVICE), val_dl)
        gap = max(abs(a - b) for a, b in zip(ref, same))
        assert gap < 1e-4, f"g=12 conversion does not reproduce B0 (gap {gap})"
        for g in GROUPS:
            model = to_gqa(load_model(b0), g)
            converted = evaluate(yva, predict(model.to(DEVICE), val_dl))
            run = train("bert", seed, model=model, epochs=UPTRAIN_EPOCHS, tag=f"bert_gqa{g}_s{seed}", out_sub="t4")
            reloaded = predict(load_model(ROOT / "checkpoints" / f"bert_gqa{g}_s{seed}").to(DEVICE), val_dl)
            assert abs(evaluate(yva, reloaded)["macro_f1"] - run["best"]["macro_f1"]) < 1e-9, "reload mismatch"
            out["runs"][f"g{g}_s{seed}"] = {"converted_no_training": converted, "uptrained": run["best"],
                                            "uptrain_sec": run["total_sec"]}
            print(f"g={g} s{seed}: converted F1 {converted['macro_f1']:.4f} -> uptrained {run['best']['macro_f1']:.4f}")
        out["runs"][f"g12_s{seed}"] = {"b0_gap_after_g12_conversion": gap, "uptrained": evaluate(yva, ref)}

    model = load_model(ROOT / "checkpoints" / f"bert_s{latency_seed}")
    L, d = config_dims(model.config)
    lens = [len(ids) for ids in AutoTokenizer.from_pretrained(MODELS["bert"])(Xva)["input_ids"]]
    out["flops_vs_full_actual_len"] = {g: sum(gqa_flops(n, L, d, g) for n in lens) / sum(gqa_flops(n, L, d, 12) for n in lens)
                                       for g in [12] + GROUPS}
    sample = [tok(t, return_tensors="pt") for t in Xva[:n_latency]]
    for g in [12] + GROUPS:
        path = ROOT / "checkpoints" / (f"bert_s{latency_seed}" if g == 12 else f"bert_gqa{g}_s{latency_seed}")
        model = load_model(path).eval()
        meas, theo = measured_flops(model, 16), gqa_flops(16, L, d, g)
        assert abs(meas - theo) / theo < 0.01, f"g={g}: measured {meas} vs theory {theo}"
        lat = {}
        for device in ["mps", "cpu"]:
            threads = torch.get_num_threads()
            if device == "cpu":
                torch.set_num_threads(1)
            model.to(device)

            def hf(enc, device=device):
                with torch.inference_mode():
                    model(**{k: v.to(device) for k, v in enc.items()})

            lat[device if device != "cpu" else "cpu_1thread"] = latency(hf, sample, device)
            torch.set_num_threads(threads)
        out["latency"][f"g{g}"] = lat
        print(f"g={g}", {dv: round(v["mean_ms"], 2) for dv, v in lat.items()})

    summary = {}
    for g in [12] + GROUPS:
        rs = [out["runs"][f"g{g}_s{s}"] for s in seeds]
        summary[g] = {m: statistics.mean(r["uptrained"][m] for r in rs) for m in ["macro_f1", "attack_recall", "benign_fpr"]}
        summary[g]["macro_f1_std"] = statistics.stdev(r["uptrained"]["macro_f1"] for r in rs) if len(rs) > 1 else 0.0
        if g != 12:
            summary[g]["converted_macro_f1"] = statistics.mean(r["converted_no_training"]["macro_f1"] for r in rs)
    out["summary_mean_over_seeds"] = summary
    (RESULTS / "t4_gqa.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["b0", "exit", "prune", "gqa"], default="b0")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2026])
    ap.add_argument("--name", default="t2_exit", help="exit stage output name (results/<name>.json/.png)")
    args = ap.parse_args()
    if args.stage == "exit":
        early_exit(args.models, args.seeds, name=args.name)
    elif args.stage == "prune":
        token_pruning(args.models, args.seeds)
    elif args.stage == "gqa":
        gqa(args.seeds)  # BERT only
    else:
        for key in args.models:
            for seed in args.seeds:
                train(key, seed)
        summarize()
