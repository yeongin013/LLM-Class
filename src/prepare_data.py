"""Stage 1: clean raw splits into data/clean/ and record stats in results/data_stats.json.

Rules (see PLAN.md, stage 1):
  - train: drop duplicate texts, drop texts that appear with both labels,
           drop texts that also appear in validation or test
  - validation: unchanged (used for all tuning)
  - test_clean: test minus texts that appear in validation or train
Texts are compared after lowercasing and collapsing whitespace (the model is uncased).
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "prompt-injection-dataset"
OUT = ROOT / "data" / "clean"
RESULTS = ROOT / "results"


def norm(text):
    return " ".join(text.lower().split())


def load(split):
    with open(RAW / f"{split}.csv", newline="") as f:
        return [{"text": r["text"], "label": int(r["label"])} for r in csv.DictReader(f)]


def save(rows, name):
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        w.writerows(rows)


def stats(rows):
    c = Counter(r["label"] for r in rows)
    return {"rows": len(rows), "label_0": c[0], "label_1": c[1]}


def dedup(rows):
    seen, out = set(), []
    for r in rows:
        k = norm(r["text"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def main():
    raw = {s: load(s) for s in ["train", "validation", "test"]}
    keys = {s: {norm(r["text"]) for r in rows} for s, rows in raw.items()}

    labels = defaultdict(set)
    for rows in raw.values():
        for r in rows:
            labels[norm(r["text"])].add(r["label"])
    conflicts = {k for k, v in labels.items() if len(v) > 1}

    train = dedup(raw["train"])
    n_dedup = len(train)
    train = [r for r in train if norm(r["text"]) not in conflicts]
    n_noconflict = len(train)
    held_out = keys["validation"] | keys["test"]
    train = [r for r in train if norm(r["text"]) not in held_out]

    validation = raw["validation"]
    test_clean = [r for r in raw["test"]
                  if norm(r["text"]) not in keys["validation"] | {norm(r["text"]) for r in train}]

    # Self-check: no duplicates inside or across the cleaned splits.
    clean = {"train": train, "validation": validation, "test_clean": test_clean}
    ck = {s: [norm(r["text"]) for r in rows] for s, rows in clean.items()}
    for s, k in ck.items():
        assert len(k) == len(set(k)), f"duplicates left in {s}"
    for a, b in [("train", "validation"), ("train", "test_clean"), ("validation", "test_clean")]:
        assert not set(ck[a]) & set(ck[b]), f"overlap left between {a} and {b}"
    assert not conflicts & set(ck["train"]), "conflicting label left in train"

    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    for name, rows in clean.items():
        save(rows, name)

    report = {
        "raw": {s: stats(rows) for s, rows in raw.items()},
        "clean": {s: stats(rows) for s, rows in clean.items()},
        "removed_from_train": {
            "duplicates": len(raw["train"]) - n_dedup,
            "label_conflicts": n_dedup - n_noconflict,
            "overlap_with_validation_or_test": n_noconflict - len(train),
        },
        "removed_from_test": len(raw["test"]) - len(test_clean),
        "conflicting_texts": sorted(conflicts),
    }
    (RESULTS / "data_stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
