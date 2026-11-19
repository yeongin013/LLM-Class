"""Stage 2: TF-IDF + LogisticRegression baseline on the cleaned data (also cascade stage 1 later).

Run: python src/tfidf.py
Writes results/tfidf.json (metrics, chosen C, latency) and results/tfidf_probs.csv (attack probability per row).
C is chosen on validation macro F1 only. test_clean is scored because TF-IDF is the reference baseline
(PLAN.md stage 2); BERT variants are scored on test_clean only once, in stage 5.
"""
import csv
import json

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union

from bench import CLEAN, RESULTS, evaluate, latency

C_GRID = [0.5, 1, 2, 5, 10]


def load(split):
    with open(CLEAN / f"{split}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [int(r["label"]) for r in rows]


def build(C):
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
    )
    return make_pipeline(features, LogisticRegression(C=C, max_iter=2000))


def main():
    data = {s: load(s) for s in ["train", "validation", "test_clean"]}
    Xtr, ytr = data["train"]
    Xva, yva = data["validation"]

    search = {}
    for C in C_GRID:
        model = build(C).fit(Xtr, ytr)
        search[C] = evaluate(yva, model.predict_proba(Xva)[:, 1])["macro_f1"]
    best_C = max(search, key=search.get)
    model = build(best_C).fit(Xtr, ytr)

    probs = {s: model.predict_proba(X)[:, 1] for s, (X, _) in data.items()}
    metrics = {s: evaluate(data[s][1], probs[s]) for s in ["validation", "test_clean"]}

    # Latency on CPU: batch 1 per sentence, and one batch call over all of validation.
    lat_b1 = latency(lambda t: model.predict_proba([t]), Xva[:500], "cpu")
    lat_all = latency(lambda X: model.predict_proba(X), [Xva] * 5, "cpu", warmup=1)
    report = {
        "C_search_validation_macro_f1": search,
        "best_C": best_C,
        "metrics": metrics,
        "latency_cpu": {
            "batch1": lat_b1,
            "batch_all_validation_per_sentence_ms": lat_all["median_ms"] / len(Xva),
        },
    }
    (RESULTS / "tfidf.json").write_text(json.dumps(report, indent=2))

    with open(RESULTS / "tfidf_probs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "row", "label", "prob_attack"])
        for s, (_, y) in data.items():
            w.writerows((s, i, label, f"{p:.6f}") for i, (label, p) in enumerate(zip(y, probs[s])))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
