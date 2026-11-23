"""Stages 4-5 (PLAN.md): TF-IDF -> BERT cascade, then the final tune / check / test evaluation.

Run: python src/final.py cascade   stage 4: forward-ratio sweep on all of validation (tau from T2), measured latency
     python src/final.py final     stage 5: choose tau, forward ratio and decision threshold on half of validation
                                   ("tune"), compare every candidate on the other half ("check"), then score B0 and
                                   2-3 finalists once on test_clean with paired bootstrap + McNemar, and dump errors.
A cascade answers with TF-IDF when its confidence max(p, 1-p) >= t, otherwise runs the BERT side (with early exit).
"""
import collections
import csv
import json
import math
import random
import re
import statistics
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

import tfidf
from bench import FIXED_LEN, RESULTS, ROOT, evaluate, latency
from model import DEVICE, EarlyExit, TAUS, exit_flops, layer_cls, load, load_model, simulate_exit

FAMILIES = ["bert", "distilbert", "bert_gqa1"]
SEEDS = [13, 42, 2026]
LAT_SEED, N_LAT, F1_DROP = 13, 500, 0.005
SWEEP_Q = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]  # stage 4 forward ratios (share of sentences sent to BERT)
TUNE_Q = [i / 40 for i in range(41)]
THETAS = [round(0.05 * i, 2) for i in range(1, 20)]
OUT = RESULTS / "final"


# ---------------------------------------------------------------- shared pieces

def tfidf_model():
    Xtr, ytr = load("train")
    model = tfidf.build(json.loads((RESULTS / "tfidf.json").read_text())["best_C"]).fit(Xtr, ytr)
    with open(RESULTS / "tfidf_probs.csv", newline="") as f:
        saved = [float(r["prob_attack"]) for r in csv.DictReader(f) if r["split"] == "validation"]
    gap = np.abs(model.predict_proba(load("validation")[0])[:, 1] - saved).max()
    assert gap < 1e-5, f"refit TF-IDF does not reproduce results/tfidf_probs.csv (gap {gap})"
    return model


def exit_matrix(run, split):
    """(L, N) attack probability at every exit (last row = the model's own head), cached per checkpoint."""
    path = ROOT / "checkpoints" / run
    cache = path / f"P_{split}.pt"
    if cache.exists():
        return torch.load(cache)
    tok, model = AutoTokenizer.from_pretrained(path), load_model(path).to(DEVICE)
    F, full = layer_cls(model, tok, load(split)[0])
    heads = torch.load(path / "exit_heads.pt")
    P = torch.stack([torch.sigmoid(F[i] @ w + b) for i, (w, b) in enumerate(heads)] + [torch.tensor(full)])
    torch.save(P, cache)
    return P


def flops(family, n, depth, L, d):
    if depth == 0:
        return 0  # answered by TF-IDF (sparse dot product, ~1e4 FLOPs: negligible)
    f = exit_flops(n, depth, L, d)
    if "gqa" in family:
        g = int(family.split("gqa")[1])
        f -= 2 * depth * 2 * n * d * d * (1 - g / 12)  # K, V projections shrink to g/12
    return f


def threshold(conf, q):
    """TF-IDF confidence cut so that a share q of `conf` is forwarded (conf < t)."""
    if q <= 0:
        return 0.0
    if q >= 1:
        return math.inf
    return float(np.quantile(conf, q))


def run_config(P, tfp, tau, t):
    """Per-sentence attack probability, BERT depth (0 = TF-IDF answered) and forwarded mask."""
    probs, depth = simulate_exit(P, tau)
    fwd = np.maximum(tfp, 1 - tfp) < t
    return np.where(fwd, probs, tfp), np.where(fwd, depth, 0), fwd


def pipeline(tf, tok, model, heads, tau, t, device):
    """Real batch-1 path: TF-IDF first (unless t = inf), BERT with early exit only when TF-IDF is unsure."""
    ee = EarlyExit(model, heads, tau, device)

    def fn(text):
        if t < math.inf:
            p = tf.predict_proba([text])[0, 1]
            if max(p, 1 - p) >= t:
                return p
        return ee(tok(text, return_tensors="pt"))[0]
    return fn


def on_devices(make_fn, texts):
    res = {}
    for device in ["mps", "cpu"]:
        threads = torch.get_num_threads()
        if device == "cpu":
            torch.set_num_threads(1)
        res["mps" if device == "mps" else "cpu_1thread"] = latency(make_fn(device), texts, device)
        torch.set_num_threads(threads)
    return res


def ref_makers():
    """B0 BERT as a plain HF forward: fixed 128 padding (the PLAN baseline) and dynamic padding."""
    path = ROOT / "checkpoints" / f"bert_s{LAT_SEED}"
    tok, model = AutoTokenizer.from_pretrained(path), load_model(path).eval()

    def hf(pad):
        def make(device):
            model.to(device)

            def fn(text):
                kw = {"padding": "max_length", "max_length": FIXED_LEN, "truncation": True} if pad else {}
                enc = tok(text, return_tensors="pt", **kw)
                with torch.inference_mode():
                    return model(**{k: v.to(device) for k, v in enc.items()}).logits
            return fn
        return make
    return {"b0_fixed128": hf(True), "b0_dynamic": hf(False)}


def references(texts, tf):
    """B0 references (see ref_makers) and TF-IDF alone."""
    return {k: on_devices(make, texts) for k, make in ref_makers().items()} | {
        "tfidf_only": {"cpu_1thread": latency(lambda x: tf.predict_proba([x]), texts, "cpu")}}


def side(family, seed):
    """Loaded BERT side for latency: tokenizer, model, exit heads."""
    path = ROOT / "checkpoints" / f"{family}_s{seed}"
    return AutoTokenizer.from_pretrained(path), load_model(path).eval(), torch.load(path / "exit_heads.pt")


def seed_latency(family, kind, params, texts, tf):
    """Measured pipeline latency for each seed's own tuned setting, plus the mean over seeds (forward shares
    differ a lot between seeds, so one seed's latency is not representative)."""
    per = {}
    for seed in SEEDS:
        tok, model, heads = side(family, seed)
        cfg = params[family, kind, seed]
        per[seed] = on_devices(lambda dv: pipeline(tf, tok, model, heads, cfg["tau"], cfg["t"], dv), texts)
    return {dv: {"mean_ms": statistics.mean(per[s][dv]["mean_ms"] for s in SEEDS)} for dv in per[SEEDS[0]]} | {
        "per_seed": per}


def t2_tau(family, seed):
    name = "t2_exit_gqa1" if "gqa" in family else "t2_exit"
    return json.loads((RESULTS / f"{name}.json").read_text())["exit"][f"{family}_s{seed}"]["chosen_tau"]


# ---------------------------------------------------------------- stage 4

def cascade():
    Xva, yva = load("validation")
    tf = tfidf_model()
    tfp = tf.predict_proba(Xva)[:, 1]
    conf = np.maximum(tfp, 1 - tfp)
    out = {"forward_ratios": SWEEP_Q, "tfidf_only": evaluate(yva, tfp), "sweep": {}, "latency": {}}
    for family in FAMILIES:
        tok = AutoTokenizer.from_pretrained(ROOT / "checkpoints" / f"{family}_s{SEEDS[0]}")
        lens = [len(ids) for ids in tok(Xva)["input_ids"]]
        rows = {q: [] for q in SWEEP_Q}
        for seed in SEEDS:
            P, tau = exit_matrix(f"{family}_s{seed}", "validation"), t2_tau(family, seed)
            L, d = P.shape[0], 768  # all three families have d = 768
            b0_flops = sum(flops("bert", n, 12, 12, 768) for n in lens)
            for q in SWEEP_Q:
                p, depth, fwd = run_config(P, tfp, tau, threshold(conf, q))
                rows[q].append(evaluate(yva, p) | {"tau": tau, "forwarded": float(fwd.mean()),
                               "flops_vs_b0": sum(flops(family, n, int(k), L, d) for n, k in zip(lens, depth)) / b0_flops})
        out["sweep"][family] = {q: {m: statistics.mean(r[m] for r in rs) for m in rs[0]} for q, rs in rows.items()}
        out["sweep"][family]["macro_f1_std"] = {q: statistics.stdev(r["macro_f1"] for r in rs) for q, rs in rows.items()}

        tok, model, heads = side(family, LAT_SEED)
        tau = t2_tau(family, LAT_SEED)
        out["latency"][family] = {q: on_devices(lambda dv, t=threshold(conf, q): pipeline(tf, tok, model, heads, tau, t, dv),
                                                Xva[:N_LAT]) for q in SWEEP_Q}
        print(family, {q: (round(out["sweep"][family][q]["macro_f1"], 4),
                           round(out["latency"][family][q]["mps"]["mean_ms"], 2)) for q in SWEEP_Q})
    out["references"] = references(Xva[:N_LAT], tf)
    OUT.mkdir(exist_ok=True)
    (OUT / "cascade_validation.json").write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------- stage 5

def split_validation(y, seed=0):
    """Stratified 50/50 split of validation into tune / check (row indices)."""
    rng = random.Random(seed)
    tune = []
    for label in (0, 1):
        rows = [i for i, v in enumerate(y) if v == label]
        rng.shuffle(rows)
        tune += rows[:len(rows) // 2]
    tune = sorted(tune)
    return tune, sorted(set(range(len(y))) - set(tune))


def best_f1(y, p):
    return max(evaluate(y, p, th)["macro_f1"] for th in THETAS)


def choose(P, tfp, y, kind, ref_f1):
    """Pick tau, forward cut t and decision threshold on the tune rows. Every setting is scored at its own best
    decision threshold. tau: fewest mean layers with tune macro F1 >= B0's (also at its best threshold) - F1_DROP.
    Forward share: the one with the highest tune macro F1 (ties -> smaller share). A "cheapest share within F1_DROP"
    rule picks 0 (TF-IDF alone) because TF-IDF is within 0.5%p of B0 on validation, yet stage 4 shows sending
    5-20% to BERT beats both. Finally the decision threshold that maximizes tune macro F1."""
    conf = np.maximum(tfp, 1 - tfp)
    tau, t = 1.01, math.inf
    if kind != "full":
        ok = [tau for tau in TAUS if best_f1(y, run_config(P, tfp, tau, math.inf)[0]) >= ref_f1 - F1_DROP]
        tau = min(ok, key=lambda v: statistics.mean(simulate_exit(P, v)[1])) if ok else 1.01
    if kind == "cascade":
        q = max(TUNE_Q, key=lambda q: (best_f1(y, run_config(P, tfp, tau, threshold(conf, q))[0]), -q))
        t = threshold(conf, q)
    p = run_config(P, tfp, tau, t)[0]
    theta = max(THETAS, key=lambda th: evaluate(y, p, th)["macro_f1"])
    return {"tau": tau, "t": t, "theta": theta}


def final():
    import scipy.stats

    Xva, yva = load("validation")
    Xte, yte = load("test_clean")
    yva, yte = np.array(yva), np.array(yte)
    tune, check = split_validation(yva.tolist())
    tf = tfidf_model()
    tfv, tft = tf.predict_proba(Xva)[:, 1], tf.predict_proba(Xte)[:, 1]
    kinds = ["full", "exit", "cascade"]
    cands = [(f, k) for f in FAMILIES for k in kinds]
    out = {"split": {"tune": len(tune), "check": len(check), "seed": 0}, "rule": choose.__doc__, "params": {},
           "check": {}, "latency_check": {}, "test": {}, "latency_test": {}, "significance": {}}

    lens_va = [len(ids) for ids in AutoTokenizer.from_pretrained(MODELS_TOK)(Xva)["input_ids"]]
    lens_te = [len(ids) for ids in AutoTokenizer.from_pretrained(MODELS_TOK)(Xte)["input_ids"]]
    params, preds = {}, {}  # (family, kind, seed) -> params / test predictions
    for seed in SEEDS:
        Pb = exit_matrix(f"bert_s{seed}", "validation").numpy()
        ref_f1 = best_f1(yva[tune], Pb[-1][tune])
        for family, kind in cands:
            P = exit_matrix(f"{family}_s{seed}", "validation")
            cfg = choose(P[:, tune], tfv[tune], yva[tune], kind, ref_f1)
            params[family, kind, seed] = cfg
            p, depth, fwd = run_config(P[:, check], tfv[check], cfg["tau"], cfg["t"])
            L = P.shape[0]
            fl = sum(flops(family, lens_va[i], int(k), L, 768) for i, k in zip(check, depth))
            b0 = sum(flops("bert", lens_va[i], 12, 12, 768) for i in check)
            out["check"].setdefault(f"{family}/{kind}", []).append(
                evaluate(yva[check], p, cfg["theta"]) | {"seed": seed, "forwarded": float(fwd.mean()),
                                                         "mean_depth": float(depth[fwd].mean()) if fwd.any() else 0.0,
                                                         "flops_vs_b0": fl / b0})
    out["params"] = {f"{f}/{k}/s{s}": v for (f, k, s), v in params.items()}
    out["check"]["tfidf_only"] = [evaluate(yva[check], tfv[check])]
    summary = {c: {m: statistics.mean(r[m] for r in rs) for m in ["macro_f1", "attack_recall", "benign_fpr"]
                   + (["forwarded", "flops_vs_b0"] if c != "tfidf_only" else [])} for c, rs in out["check"].items()}

    texts = [Xva[i] for i in check[:N_LAT]]
    for family, kind in cands:
        out["latency_check"][f"{family}/{kind}"] = seed_latency(family, kind, params, texts, tf)
    out["latency_check"] |= references(texts, tf)
    for c in summary:
        summary[c]["mps_mean_ms"] = out["latency_check"][c].get("mps", {}).get("mean_ms")
        summary[c]["cpu_mean_ms"] = out["latency_check"][c]["cpu_1thread"]["mean_ms"]
    out["check_summary"] = summary

    # Finalists: up to 3 fastest (MPS mean) candidates within F1_DROP of B0 on check.
    b0 = summary["bert/full"]["macro_f1"]
    ok = [c for c in summary if c not in ("bert/full", "tfidf_only") and summary[c]["macro_f1"] >= b0 - F1_DROP]
    finalists = sorted(ok, key=lambda c: summary[c]["mps_mean_ms"])[:3]
    assert finalists, "no candidate within F1_DROP of B0 on check"
    out["finalists"] = finalists
    print("check summary", json.dumps(summary, indent=1), "finalists", finalists)

    # ---- test_clean: scored once, B0 and finalists only.
    for c in ["bert/full"] + finalists:
        family, kind = c.split("/")
        for seed in SEEDS:
            cfg = params[family, kind, seed]
            P = exit_matrix(f"{family}_s{seed}", "test_clean")
            p, depth, fwd = run_config(P, tft, cfg["tau"], cfg["t"])
            preds[c, seed] = (p >= cfg["theta"]).astype(int)
            L = P.shape[0]
            out["test"].setdefault(c, []).append(evaluate(yte, p, cfg["theta"]) | {
                "seed": seed, "forwarded": float(fwd.mean()), "macro_f1_at_0.5": evaluate(yte, p)["macro_f1"],
                "flops_vs_b0": sum(flops(family, n, int(k), L, 768) for n, k in zip(lens_te, depth))
                / sum(flops("bert", n, 12, 12, 768) for n in lens_te)})
            if c == finalists[0] and seed == LAT_SEED:
                b0p = exit_matrix(f"bert_s{seed}", "test_clean")[-1].numpy()
                with open(OUT / "errors_test.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["row", "label", "pred", "prob", "answered_by", "bert_depth", "tfidf_prob", "b0_prob", "text"])
                    for i in np.where(preds[c, seed] != yte)[0]:
                        w.writerow([i, yte[i], preds[c, seed][i], f"{p[i]:.4f}", "bert" if fwd[i] else "tfidf",
                                    int(depth[i]), f"{tft[i]:.4f}", f"{b0p[i]:.4f}", Xte[i]])
        out["latency_test"][c] = seed_latency(family, kind, params, Xte[:N_LAT], tf)
    out["latency_test"] |= references(Xte[:N_LAT], tf)
    out["test"]["tfidf_only"] = [evaluate(yte, tft)]

    # ---- significance vs B0 on test_clean: paired bootstrap of the seed-mean macro F1 gap, McNemar per seed.
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    boots = rng.integers(0, len(yte), size=(2000, len(yte)))
    for c in finalists:
        gaps = [np.mean([f1_score(yte[b], preds[c, s][b], average="macro") - f1_score(yte[b], preds["bert/full", s][b],
                         average="macro") for s in SEEDS]) for b in boots]
        mc = []
        for s in SEEDS:
            a, r = preds[c, s] == yte, preds["bert/full", s] == yte
            only_a, only_r = int((a & ~r).sum()), int((~a & r).sum())
            mc.append({"seed": s, "only_candidate_right": only_a, "only_b0_right": only_r,
                       "p": scipy.stats.binomtest(only_a, only_a + only_r).pvalue if only_a + only_r else 1.0})
        out["significance"][c] = {"f1_gap_mean": float(np.mean([out["test"][c][i]["macro_f1"] - out["test"]["bert/full"][i]
                                                               ["macro_f1"] for i in range(len(SEEDS))])),
                                  "bootstrap_95ci": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
                                  "mcnemar": mc}
    out["test_summary"] = {c: {m: statistics.mean(r[m] for r in rs) for m in ["macro_f1", "attack_recall", "benign_fpr"]
                               + (["macro_f1_at_0.5", "forwarded", "flops_vs_b0"] if c != "tfidf_only" else [])}
                           for c, rs in out["test"].items()}
    (OUT / "final.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({"test": out["test_summary"], "sig": out["significance"]}, indent=1))
    plot_pareto(summary)


MODELS_TOK = "bert-base-uncased"  # all three families share the uncased WordPiece vocabulary


def plot_pareto(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    colors = {"bert": "#2a78d6", "distilbert": "#eb6834", "bert_gqa1": "#1baf7a"}  # categorical slots 1-3, validated
    markers = {"full": "o", "exit": "s", "cascade": "^"}
    ink, muted, surface = "#1f1f1e", "#6b6a64", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=surface)
    for c, s in summary.items():
        if c == "tfidf_only":
            continue
        family, kind = c.split("/")
        ax.scatter(s["mps_mean_ms"], s["macro_f1"], color=colors[family], marker=markers[kind], s=64,
                   edgecolor=surface, linewidth=2, zorder=3)
        ax.annotate(c, (s["mps_mean_ms"], s["macro_f1"]), xytext=(6, 4), textcoords="offset points", color=ink, fontsize=8)
    b0 = summary["bert/full"]["macro_f1"]
    ax.axhline(b0 - F1_DROP, color=muted, lw=1, ls="--")
    ax.annotate("B0 - 0.5%p", (ax.get_xlim()[1], b0 - F1_DROP), xytext=(-4, 4), textcoords="offset points",
                ha="right", color=muted, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("mean latency per sentence, MPS batch 1 (ms, log scale)", color=ink)
    ax.set_ylabel("macro F1 (validation check half, mean of 3 seeds)", color=ink)
    ax.set_title("Accuracy vs latency (circle = full, square = early exit, triangle = cascade)", color=ink,
                 fontsize=10, loc="left")
    ax.set_facecolor(surface)
    ax.grid(color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=muted)
    fig.tight_layout()
    fig.savefig(OUT / "pareto_check.png", dpi=150)


# ---------------------------------------------------------------- robustness checks (report.md section 5)

KINDS = ["full", "exit", "cascade"]


def saved_params():
    out = json.loads((OUT / "final.json").read_text())
    params = {}
    for key, v in out["params"].items():
        family, kind, seed = key.split("/")
        params[family, kind, int(seed[1:])] = v
    return out, params


def latency_repeat(rounds=5):
    """Re-measure the stage 5 test latencies `rounds` times, shuffling the order of (config, seed) units and of
    the two devices every round, so warm-up order and heat cannot favor one config."""
    out, params = saved_params()
    texts = load("test_clean")[0][:N_LAT]
    tf = tfidf_model()
    makers, loaded = ref_makers(), {}
    units = [(c, None) for c in makers] + [("tfidf_only", None)] + [(c, s) for c in out["finalists"] for s in SEEDS]
    rng = random.Random(0)
    runs = {}
    for r in range(rounds):
        rng.shuffle(units)
        for c, seed in units:
            for device in (["cpu"] if c == "tfidf_only" else rng.sample(["mps", "cpu"], 2)):
                if c == "tfidf_only":
                    def make(dv):
                        return lambda x: tf.predict_proba([x])
                elif seed is None:
                    make = makers[c]
                else:
                    family, kind = c.split("/")
                    if (family, seed) not in loaded:
                        loaded[family, seed] = side(family, seed)
                    tok, model, heads = loaded[family, seed]
                    cfg = params[family, kind, seed]

                    def make(dv, tok=tok, model=model, heads=heads, cfg=cfg):
                        return pipeline(tf, tok, model, heads, cfg["tau"], cfg["t"], dv)
                threads = torch.get_num_threads()
                if device == "cpu":
                    torch.set_num_threads(1)
                ms = latency(make(device), texts, device)["mean_ms"]
                torch.set_num_threads(threads)
                runs.setdefault(c, {}).setdefault(device, {}).setdefault(r, []).append(ms)
        print("round", r + 1, "done")
    summary = {}
    for c, by_dev in runs.items():
        for device, by_round in by_dev.items():
            vals = [statistics.mean(v) for v in by_round.values()]  # mean over seeds within a round
            summary.setdefault(c, {})["mps" if device == "mps" else "cpu_1thread"] = {
                "mean_ms": statistics.mean(vals), "sd_ms": statistics.stdev(vals), "min_ms": min(vals),
                "max_ms": max(vals), "rounds": vals}
    for c in out["finalists"]:
        summary[c]["speedup_vs_b0"] = {
            dev: {ref: summary[ref][dev]["mean_ms"] / summary[c][dev]["mean_ms"] for ref in ["b0_dynamic", "b0_fixed128"]}
            for dev in ["mps", "cpu_1thread"]}
    (OUT / "latency_repeat.json").write_text(json.dumps({"rounds": rounds, "texts": len(texts), "summary": summary},
                                                        indent=2))
    print(json.dumps({c: {d: (round(v["mean_ms"], 2), round(v["sd_ms"], 2)) for d, v in s.items() if "mean_ms" in v}
                      for c, s in summary.items()}, indent=1))


def resplit(n_splits=20, families=FAMILIES, name="resplit"):
    """Repeat the stage 5 selection over `n_splits` random tune/check splits of validation (split 0 = the one in
    final.json). Reports how chosen settings, check F1 and finalist membership vary with the split, and - as a
    sensitivity analysis only, nothing is selected from it - how test_clean F1 varies."""
    Xva, yva = load("validation")
    Xte, yte = load("test_clean")
    yva, yte = np.array(yva), np.array(yte)
    tf = tfidf_model()
    tfv, tft = tf.predict_proba(Xva)[:, 1], tf.predict_proba(Xte)[:, 1]
    lat = {c: v["mps_mean_ms"] for c, v in json.loads((OUT / "final.json").read_text())["check_summary"].items()}
    cands = [f"{f}/{k}" for f in families for k in KINDS]
    P = {(f, s, sp): exit_matrix(f"{f}_s{s}", sp) for f in set(families) | {"bert"} for s in SEEDS
         for sp in ["validation", "test_clean"]}
    per_split = []
    for sp in range(n_splits):
        tune, check = split_validation(yva.tolist(), seed=sp)
        res = {c: [] for c in cands}
        for seed in SEEDS:
            ref = best_f1(yva[tune], P["bert", seed, "validation"][-1].numpy()[tune])
            for c in cands:
                family, kind = c.split("/")
                Pv, Pt = P[family, seed, "validation"], P[family, seed, "test_clean"]
                cfg = choose(Pv[:, tune], tfv[tune], yva[tune], kind, ref)
                pc = run_config(Pv[:, check], tfv[check], cfg["tau"], cfg["t"])[0]
                pt, _, fwd = run_config(Pt, tft, cfg["tau"], cfg["t"])
                res[c].append(cfg | {"check_f1": evaluate(yva[check], pc, cfg["theta"])["macro_f1"],
                                     "test_f1": evaluate(yte, pt, cfg["theta"])["macro_f1"],
                                     "test_forwarded": float(fwd.mean())})
        mean = {c: {m: statistics.mean(r[m] for r in rs) for m in ["check_f1", "test_f1", "test_forwarded"]}
                for c, rs in res.items()}
        b0 = mean["bert/full"]["check_f1"]
        ok = [c for c in cands if c != "bert/full" and c in lat and mean[c]["check_f1"] >= b0 - F1_DROP]
        per_split.append({"split": sp, "runs": res, "mean": mean, "finalists": sorted(ok, key=lambda c: lat[c])[:3]})
        print("split", sp, per_split[-1]["finalists"])

    summary = {}
    for c in cands:
        cf = [s["mean"][c]["check_f1"] for s in per_split]
        tf1 = [s["mean"][c]["test_f1"] for s in per_split]
        runs = [r for s in per_split for r in s["runs"][c]]
        summary[c] = {
            "check_f1": {"mean": statistics.mean(cf), "sd": statistics.stdev(cf), "min": min(cf), "max": max(cf)},
            "test_f1": {"mean": statistics.mean(tf1), "sd": statistics.stdev(tf1), "min": min(tf1), "max": max(tf1)},
            "test_f1_minus_b0": statistics.mean(s["mean"][c]["test_f1"] - s["mean"]["bert/full"]["test_f1"] for s in per_split),
            "finalist_share": sum(c in s["finalists"] for s in per_split) / n_splits,
            "chosen_theta": dict(sorted(collections.Counter(r["theta"] for r in runs).items())),
            "chosen_tau": dict(sorted(collections.Counter(r["tau"] for r in runs).items())),
            "test_forwarded": {"mean": statistics.mean(r["test_forwarded"] for r in runs),
                               "min": min(r["test_forwarded"] for r in runs), "max": max(r["test_forwarded"] for r in runs)}}
    (OUT / f"{name}.json").write_text(json.dumps({"n_splits": n_splits, "summary": summary, "per_split": per_split},
                                                 indent=2, default=str))
    print(json.dumps({c: {"check": round(v["check_f1"]["mean"], 4), "test": round(v["test_f1"]["mean"], 4),
                          "test_sd": round(v["test_f1"]["sd"], 4), "finalist": v["finalist_share"]}
                      for c, v in summary.items()}, indent=1))


EXPLICIT = re.compile(r"\b(ignore|disregard|forget|override|bypass|jailbreak|pretend|dan|developer mode|"
                      r"previous instructions|system prompt|no restrictions|unrestricted)\b", re.I)


def short_attacks(max_words=8):
    """How common are short attacks without explicit override words in each split, and how well do TF-IDF and B0
    (threshold 0.5, mean of 3 seeds) catch them? EXPLICIT is a hand-made keyword heuristic."""
    tf = tfidf_model()
    out = {"max_words": max_words, "explicit_pattern": EXPLICIT.pattern, "splits": {}}
    for split in ["train", "validation", "test_clean"]:
        X, y = load(split)
        y = np.array(y)
        short = np.array([len(t.split()) <= max_words for t in X])
        implicit = short & np.array([not EXPLICIT.search(t) for t in X])
        att = y == 1
        rec = {"attacks": int(att.sum()), "short_attacks": int((att & short).sum()),
               "short_implicit_attacks": int((att & implicit).sum()),
               "short_implicit_share_of_attacks": float((att & implicit).sum() / att.sum()),
               "short_implicit_benign": int((~att & implicit).sum())}
        if split != "train":
            tfp = tf.predict_proba(X)[:, 1]
            b0 = np.mean([exit_matrix(f"bert_s{s}", split)[-1].numpy() >= 0.5 for s in SEEDS], axis=0)
            for nm, mask in [("all_attacks", att), ("short_implicit_attacks", att & implicit),
                             ("other_attacks", att & ~implicit)]:
                rec[f"recall_{nm}"] = {"tfidf": float((tfp[mask] >= 0.5).mean()), "b0_seed_mean": float(b0[mask].mean())}
        out["splits"][split] = rec
    (OUT / "short_attacks.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["splits"], indent=1))


if __name__ == "__main__":
    {"cascade": cascade, "final": final, "latency_repeat": latency_repeat, "resplit": resplit,
     "short_attacks": short_attacks}[sys.argv[1]]()
