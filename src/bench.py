"""Stage 2: complexity analysis, plus shared measurement helpers reused by later stages.

Run: python src/bench.py            complexity analysis (stage 2)
     python src/bench.py padding    T1 dynamic padding on trained B0 (stage 3-T1)
Writes results/complexity.json, results/latency_pilot.json, results/flops_vs_n.png,
       results/t1_padding.json.

Helpers:
  theory_flops(n, L, d)   FLOPs of one forward pass (matmuls only), see PLAN.md section 3
  measured_flops(model, n)
  latency(fn, inputs, device)   per-call median / p95 in ms
  evaluate(y, prob, threshold)  accuracy, macro F1, attack recall, benign FPR, ROC-AUC
"""
import json
import statistics
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
RESULTS = ROOT / "results"
MODELS = {"bert": "bert-base-uncased", "distilbert": "distilbert-base-uncased"}
FIXED_LEN = 128


def theory_flops(n, L, d):
    # Per layer: Q,K,V,O projections 4nd^2 + FFNN (d -> 4d -> d) 8nd^2 + QK^T and AV 2n^2 d multiply-adds.
    # Head: d x d (BERT pooler / DistilBERT pre_classifier) + d x 2 classifier. FLOPs = 2 x multiply-adds.
    return 2 * (L * (12 * n * d * d + 2 * n * n * d) + d * d + 2 * d)


def components(n, d):
    proj, ffn, attn = 4 * n * d * d, 8 * n * d * d, 2 * n * n * d
    total = proj + ffn + attn
    return {"proj": proj / total, "ffn": ffn / total, "attn_n2": attn / total}


def measured_flops(model, n):
    x = torch.randint(1000, 2000, (1, n))
    counter = FlopCounterMode(display=False)
    with torch.inference_mode(), counter:
        model(input_ids=x, attention_mask=torch.ones_like(x))
    return counter.get_total_flops()


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def latency(fn, inputs, device, warmup=20):
    """Call fn on each input once (after warm-up) and return per-call ms stats."""
    for x in inputs[:warmup]:
        fn(x)
    sync(device)
    times = []
    for x in inputs:
        t = time.perf_counter()
        fn(x)
        sync(device)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"median_ms": statistics.median(times), "mean_ms": statistics.mean(times),
            "p95_ms": times[int(0.95 * (len(times) - 1))], "calls": len(times)}


def evaluate(y, prob, threshold=0.5):
    pred = [int(p >= threshold) for p in prob]
    benign = [p for t, p in zip(y, pred) if t == 0]
    attack = [p for t, p in zip(y, pred) if t == 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "attack_recall": sum(attack) / len(attack),
        "benign_fpr": sum(benign) / len(benign),
        "roc_auc": roc_auc_score(y, prob),
    }


def load_texts(split):
    import csv
    with open(CLEAN / f"{split}.csv", newline="") as f:
        return [r["text"] for r in csv.DictReader(f)]


def config_dims(cfg):
    # BertConfig and DistilBertConfig name the same things differently.
    L = getattr(cfg, "num_hidden_layers", None) or cfg.n_layers
    d = getattr(cfg, "hidden_size", None) or cfg.dim
    ffn = getattr(cfg, "intermediate_size", None) or cfg.hidden_dim
    assert ffn == 4 * d, "theory_flops assumes FFNN hidden = 4d"
    return L, d


def main():
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    RESULTS.mkdir(exist_ok=True)
    splits = ["train", "validation", "test_clean"]
    out = {"fixed_len": FIXED_LEN, "models": {}}
    pilot = {"note": "random-init weights; latency does not depend on weight values", "models": {}}
    lengths = {}

    for key, name in MODELS.items():
        tok = AutoTokenizer.from_pretrained(name)
        cfg = AutoConfig.from_pretrained(name, num_labels=2)
        L, d = config_dims(cfg)
        model = AutoModelForSequenceClassification.from_config(cfg).eval()

        lens = {s: [len(ids) for ids in tok(load_texts(s))["input_ids"]] for s in splits}
        lengths[key] = lens
        assert max(max(v) for v in lens.values()) <= FIXED_LEN

        # 1. Theory vs FlopCounterMode at representative lengths.
        checks = {}
        for n in sorted({round(statistics.mean(lens["test_clean"])), max(lens["train"]), FIXED_LEN, 512}):
            m, t = measured_flops(model, n), theory_flops(n, L, d)
            checks[n] = {"measured_gflops": m / 1e9, "theory_gflops": t / 1e9, "ratio": m / t}
        # Check on real lengths: every distinct length in validation + test_clean.
        real = sorted(set(lens["validation"]) | set(lens["test_clean"]))
        worst = max(abs(measured_flops(model, n) / theory_flops(n, L, d) - 1) for n in real)

        # 2. Component shares and dataset-level FLOPs saved by dropping padding.
        per_split = {}
        for s, v in lens.items():
            q = sorted(v)
            per_split[s] = {
                "mean_tokens": statistics.mean(v), "p95_tokens": q[int(0.95 * (len(q) - 1))], "max_tokens": q[-1],
                "pad_fraction_at_fixed": 1 - statistics.mean(v) / FIXED_LEN,
                "flops_ratio_fixed_over_actual": theory_flops(FIXED_LEN, L, d) * len(v) / sum(theory_flops(n, L, d) for n in v),
            }
        out["models"][key] = {
            "name": name, "layers": L, "d_model": d,
            "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
            "theory_vs_measured": checks,
            "worst_rel_error_on_real_lengths": worst,
            "component_share": {n: components(n, d) for n in [per_split["test_clean"]["mean_tokens"], 29, FIXED_LEN, 512]},
            "splits": per_split,
        }

        # 3. Latency pilot: batch 1, 200 validation sentences, fixed padding vs actual length.
        texts = load_texts("validation")[:200]
        fixed = [tok(t, padding="max_length", max_length=FIXED_LEN, return_tensors="pt") for t in texts]
        actual = [tok(t, return_tensors="pt") for t in texts]
        pilot["models"][key] = {}
        for device in ["mps", "cpu"]:
            threads = torch.get_num_threads()
            if device == "cpu":
                torch.set_num_threads(1)  # single thread: stable, comparable CPU numbers
            m = model.to(device)

            def run(enc, m=m, device=device):
                with torch.inference_mode():
                    m(**{k: v.to(device) for k, v in enc.items()})

            res = {"fixed_128": latency(run, fixed, device), "actual_len": latency(run, actual, device)}
            res["speedup"] = res["fixed_128"]["median_ms"] / res["actual_len"]["median_ms"]
            pilot["models"][key][device if device != "cpu" else "cpu_1thread"] = res
            torch.set_num_threads(threads)
        print(key, "done")

    (RESULTS / "complexity.json").write_text(json.dumps(out, indent=2))
    (RESULTS / "latency_pilot.json").write_text(json.dumps(pilot, indent=2))
    plot(out, lengths)
    print(json.dumps({k: {"worst_err": v["worst_rel_error_on_real_lengths"], "splits": v["splits"]} for k, v in out["models"].items()}, indent=2))
    print(json.dumps(pilot, indent=2))


def plot(out, lengths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"bert": "#2a78d6", "distilbert": "#eb6834"}  # categorical slots 1-2, validated light mode
    ink, muted, surface = "#1f1f1e", "#6b6a64", "#fcfcfb"
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, height_ratios=[3, 2], facecolor=surface)
    ns = list(range(1, 161))
    for key, m in out["models"].items():
        g = [theory_flops(n, m["layers"], m["d_model"]) / 1e9 for n in ns]
        top.plot(ns, g, color=colors[key], lw=2)
        top.annotate(f"{key} ({m['layers']} layers)", (ns[-1], g[-1]), xytext=(-4, 6), textcoords="offset points",
                     ha="right", color=ink, fontsize=9)
    top.axvline(FIXED_LEN, color=muted, lw=1, ls="--")
    top.annotate("fixed padding = 128", (FIXED_LEN, top.get_ylim()[1] * 0.45), xytext=(-6, 0), textcoords="offset points",
                 ha="right", color=muted, fontsize=9)
    top.set_ylabel("GFLOPs per sentence", color=ink)
    top.set_title("Forward-pass cost grows ~linearly with tokens; real inputs are far shorter than 128", color=ink,
                  fontsize=11, loc="left")

    v = lengths["bert"]["test_clean"]  # same WordPiece vocab, so DistilBERT lengths are identical
    bottom.hist(v, bins=range(0, max(v) + 2), color="#2a78d6", rwidth=0.85)
    bottom.set_ylabel("sentences (test_clean)", color=ink)
    bottom.set_xlabel("tokens per sentence (n)", color=ink)
    bottom.set_xlim(0, 160)
    for ax in (top, bottom):
        ax.set_facecolor(surface)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=muted)
    fig.tight_layout()
    fig.savefig(RESULTS / "flops_vs_n.png", dpi=150)


def padding_t1(seeds=(13, 42, 2026), latency_seed=13, n_batch1=500, repeats=3):
    """T1: same trained weights, three padding modes. Checks equal predictions, then measures cost."""
    import csv
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with open(CLEAN / "validation.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    texts, y = [r["text"] for r in rows], [int(r["label"]) for r in rows]
    B = 32

    def batches(tok, mode):
        """Yield (row indices, encoded batch). fixed: pad to 128; dynamic: pad to batch max; bucketed: sort by length first."""
        order = list(range(len(texts)))
        if mode == "bucketed":
            lens = [len(ids) for ids in tok(texts)["input_ids"]]
            order.sort(key=lambda i: lens[i])
        kw = {"padding": "max_length", "max_length": FIXED_LEN} if mode == "fixed" else {"padding": "longest"}
        for i in range(0, len(order), B):
            idx = order[i:i + B]
            yield idx, tok([texts[j] for j in idx], return_tensors="pt", **kw)

    def run_probs(model, tok, mode, device):
        probs = [0.0] * len(texts)
        with torch.inference_mode():
            for idx, enc in batches(tok, mode):
                p = torch.softmax(model(**{k: v.to(device) for k, v in enc.items()}).logits.float(), -1)[:, 1]
                for j, v in zip(idx, p.tolist()):
                    probs[j] = v
        return probs

    out = {"equivalence": {}, "batch1_latency": {}, "batch32_throughput": {}, "tokenize_ms_per_sentence": {}}
    modes = ["fixed", "dynamic", "bucketed"]
    for key in MODELS:
        for seed in seeds:
            path = ROOT / "checkpoints" / f"{key}_s{seed}"
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path).to("mps").eval()
            probs = {m: run_probs(model, tok, m, "mps") for m in modes}
            ref = probs["fixed"]
            out["equivalence"][f"{key}_s{seed}"] = {
                m: {"max_abs_prob_diff": max(abs(a - b) for a, b in zip(ref, probs[m])),
                    "pred_disagreements": sum((a >= 0.5) != (b >= 0.5) for a, b in zip(ref, probs[m])),
                    "macro_f1": evaluate(y, probs[m])["macro_f1"]} for m in modes}

            if seed != latency_seed:
                continue
            # Cost does not depend on weights, so one seed per model is enough.
            sample = texts[:n_batch1]
            fixed1 = [tok(t, padding="max_length", max_length=FIXED_LEN, return_tensors="pt") for t in sample]
            actual1 = [tok(t, return_tensors="pt") for t in sample]
            out["tokenize_ms_per_sentence"][key] = latency(lambda t: tok(t, return_tensors="pt"), sample, "cpu")
            out["batch1_latency"][key] = {}
            out["batch32_throughput"][key] = {}
            for device in ["mps", "cpu"]:
                threads = torch.get_num_threads()
                if device == "cpu":
                    torch.set_num_threads(1)
                m = model.to(device)

                def fwd(enc, m=m, device=device):
                    with torch.inference_mode():
                        m(**{k: v.to(device) for k, v in enc.items()})

                dev_name = device if device != "cpu" else "cpu_1thread"
                res = {"fixed_128": latency(fwd, fixed1, device), "actual_len": latency(fwd, actual1, device)}
                res["speedup"] = res["fixed_128"]["median_ms"] / res["actual_len"]["median_ms"]
                out["batch1_latency"][key][dev_name] = res

                # Throughput over all of validation; pre-tokenized so only the model is timed.
                tput = {}
                for mode in modes:
                    encs = [enc for _, enc in batches(tok, mode)]
                    fwd(encs[0])
                    times = []
                    for _ in range(repeats):
                        sync(device)
                        t = time.perf_counter()
                        for enc in encs:
                            fwd(enc)
                        sync(device)
                        times.append(time.perf_counter() - t)
                    tput[mode] = {"sentences_per_sec": len(texts) / statistics.median(times)}
                for mode in modes:
                    tput[mode]["speedup_vs_fixed"] = tput[mode]["sentences_per_sec"] / tput["fixed"]["sentences_per_sec"]
                out["batch32_throughput"][key][dev_name] = tput
                torch.set_num_threads(threads)
            model.to("mps")
        print(key, "done")

    (RESULTS / "t1_padding.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys
    padding_t1() if sys.argv[1:] == ["padding"] else main()
