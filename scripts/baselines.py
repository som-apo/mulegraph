"""Baselines: what a rulebook actually gets you.

The obvious question about any fraud model is "why not just write rules?"
Every bank starts with rules, and rules are cheaper, faster and easier to
defend to a regulator. So the honest thing is to build the rulebook, run
it on the same data, and show the gap.

A rulebook produces a set, not a ranking, so it cannot be compared at a
fixed alert budget. It is compared on its own terms: how many accounts it
flags, how many of those are mules, and how many mules it misses. The
model is then evaluated at that same queue size, which is the only
like-for-like comparison available.
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")


def load_scored():
    for ext in (".parquet", ".csv.gz"):
        hits = glob.glob(f"data/scored{ext}")
        if hits:
            return (pd.read_parquet(hits[0]) if ext == ".parquet"
                    else pd.read_csv(hits[0]))
    raise SystemExit("run scripts/run_pipeline.py first")


def _prf(flagged: np.ndarray, y: np.ndarray) -> dict:
    n = int(flagged.sum())
    tp = int((flagged & (y == 1)).sum())
    return {"flagged": n, "caught": tp,
            "precision": tp / max(1, n),
            "recall": tp / max(1, int(y.sum()))}


def rule_baselines(f: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    rows = []

    # The rule an analyst writes on day one, straight from the pattern
    # description: money comes in, leaves fast.
    naive = ((f["pass_through"] > 0.90) & (f["drain_median_h"] < 6)).to_numpy()
    r = _prf(naive, y)
    r["baseline"] = "Rule: pass-through > 0.90 and drains within 6h"
    rows.append(r)

    # The rule after a few weeks of tuning, once the shopkeepers have
    # complained: add fan-in and a changepoint condition.
    tuned = ((f["pass_through"] > 0.85)
             & (f["drain_median_h"] < 12)
             & (f["fan_in"] >= 10)
             & (f["throughput_jump"] > 3)).to_numpy()
    r = _prf(tuned, y)
    r["baseline"] = "Rule: + fan-in >= 10 and 3x throughput jump"
    rows.append(r)

    # The single strongest feature at its best threshold. This is the
    # ceiling for any one-signal system.
    best_col, best_f1, best_mask = None, -1.0, None
    for c in ["n_debits", "n_credits", "pass_through", "fan_in",
              "throughput_jump", "active_days_ratio"]:
        if c not in f.columns:
            continue
        v = f[c].to_numpy()
        for q in np.linspace(0.90, 0.999, 40):
            m = v >= np.quantile(v, q)
            s = _prf(m, y)
            f1 = (2 * s["precision"] * s["recall"] /
                  max(1e-9, s["precision"] + s["recall"]))
            if f1 > best_f1:
                best_f1, best_col, best_mask = f1, c, m
    if best_mask is not None:
        r = _prf(best_mask, y)
        r["baseline"] = f"Best single feature, best threshold ({best_col})"
        rows.append(r)

    rng = np.random.default_rng(0)
    k = int(rows[1]["flagged"]) or 200
    m = np.zeros(len(f), dtype=bool)
    m[rng.choice(len(f), size=min(k, len(f)), replace=False)] = True
    r = _prf(m, y)
    r["baseline"] = f"Random selection of {k:,} accounts"
    rows.append(r)

    return pd.DataFrame(rows)[["baseline", "flagged", "caught",
                               "precision", "recall"]]


def model_at(f: pd.DataFrame, y: np.ndarray, k: int) -> dict:
    idx = np.argsort(-f["score"].to_numpy())[:k]
    m = np.zeros(len(f), dtype=bool)
    m[idx] = True
    return _prf(m, y)


def compare(f: pd.DataFrame) -> pd.DataFrame:
    """Baselines plus the model evaluated at each baseline's queue size."""
    y = (f["true_mule"].astype(int).to_numpy() if "true_mule" in f.columns
         else f["is_mule"].astype(int).to_numpy())
    base = rule_baselines(f, y)

    rows = []
    for _, b in base.iterrows():
        k = int(b["flagged"])
        mm = model_at(f, y, k) if k > 0 else {"precision": 0, "recall": 0}
        rows.append({
            "approach": b["baseline"],
            "flagged": k,
            "caught": int(b["caught"]),
            "precision": round(float(b["precision"]), 3),
            "recall": round(float(b["recall"]), 3),
            "model precision at same size": round(float(mm["precision"]), 3),
            "model recall at same size": round(float(mm["recall"]), 3),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    f = load_scored()
    t = compare(f)
    print("\n=== Why not just write rules? ===\n")
    print(t.to_string(index=False))
    best_rule = t.iloc[t["recall"].idxmax()]
    print(f"\nThe best rulebook flags {int(best_rule['flagged']):,} accounts to "
          f"catch {best_rule['recall']:.1%} of mules at "
          f"{best_rule['precision']:.1%} precision.")
    print(f"The model, reviewing the same {int(best_rule['flagged']):,}, catches "
          f"{best_rule['model recall at same size']:.1%} at "
          f"{best_rule['model precision at same size']:.1%}.")
    print("\nRules are not stupid -- they are cheap, fast and easy to defend.")
    print("They fail because a careful mule keeps every single condition")
    print("inside the acceptable range, and the combination is what is rare.")
