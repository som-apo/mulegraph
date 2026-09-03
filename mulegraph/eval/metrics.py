"""Evaluation metrics.

Written before any model exists, deliberately. Once you have a score in
hand it is very easy to pick the metric that flatters it; writing the
scoring code first removes that temptation.

WHAT WE REFUSE TO REPORT
------------------------
Accuracy. At 0.5% prevalence, a model that says "nobody is a mule" scores
99.5%. It is not a hard number to beat, it is a meaningless one.

ROC-AUC. It flatters badly under class imbalance. 80% recall at a 2% false
positive rate reads as roughly 0.95 AUC while five out of every six alerts
an analyst opens are wrong. The false-positive axis is scaled by the huge
negative class, so it hides the thing operations actually feels.

WHAT WE REPORT
--------------
PR-AUC             ranking quality against the positive class
Precision@k        of the top k flagged, how many are real
Recall@budget      if ops reviews k/day, what share of mules do we catch
Cost curve         rupees prevented vs legitimate users disrupted
Difficulty split   blatant and subtle reported separately, always
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


# Tiered response costs. The whole argument of the project lives in this
# table: because the cheap actions are genuinely cheap, we can afford to
# be generous with recall at low thresholds.
TIER_COST = {
    "monitor": 0.0,      # customer never knows
    "limit": 50.0,       # minor reversible friction
    "review": 200.0,     # analyst time, invisible to the customer
    "hold": 2000.0,      # real harm -- reserved for high confidence
}
AVG_MULE_VALUE = 250_000.0   # rupees a mule account moves if left alone


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    idx = np.argsort(-scores)[:k]
    return float(y_true[idx].mean())


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    total = y_true.sum()
    if total == 0:
        return 0.0
    idx = np.argsort(-scores)[:k]
    return float(y_true[idx].sum() / total)


def cost_curve(y_true: np.ndarray, scores: np.ndarray,
               budgets=(200, 500, 1000, 2000, 4000)) -> pd.DataFrame:
    """Rupees prevented vs legitimate accounts disrupted, per alert budget.

    'Disrupted' counts only false positives that reach manual review or
    above -- a false positive on silent monitoring costs the customer
    nothing, which is exactly why the tiered policy is worth having.
    """
    rows = []
    for k in budgets:
        k = min(k, len(scores))
        idx = np.argsort(-scores)[:k]
        tp = int(y_true[idx].sum())
        fp = k - tp
        rows.append({
            "budget": k,
            "caught": tp,
            "false_positives": fp,
            "precision": tp / max(1, k),
            "recall": tp / max(1, y_true.sum()),
            "value_prevented": tp * AVG_MULE_VALUE,
            "review_cost": k * TIER_COST["review"],
            "net_value": tp * AVG_MULE_VALUE - k * TIER_COST["review"],
        })
    return pd.DataFrame(rows)


def evaluate(y_true, scores, difficulty=None, budget: int = 200,
             label: str = "model") -> dict:
    """Full report for one scoring run."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    res = {
        "label": label,
        "n": len(y_true),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),  # shown only to be criticised
        f"precision@{budget}": precision_at_k(y_true, scores, budget),
        f"recall@{budget}": recall_at_k(y_true, scores, budget),
    }

    # Difficulty split. Each slice is scored against the same ranking over
    # the whole population -- we do not re-rank within a slice, because ops
    # sees one queue, not two.
    if difficulty is not None:
        difficulty = np.asarray(difficulty).astype(str)
        for d in ("blatant", "subtle"):
            mask_pos = (difficulty == d) & (y_true == 1)
            if mask_pos.sum() == 0:
                continue
            keep = mask_pos | (y_true == 0)
            yk, sk = y_true[keep], scores[keep]
            res[f"{d}_pr_auc"] = float(average_precision_score(yk, sk))
            res[f"{d}_precision@{budget}"] = precision_at_k(yk, sk, budget)
            res[f"{d}_recall@{budget}"] = recall_at_k(yk, sk, budget)
    return res


def print_report(res: dict, budget: int = 200):
    print(f"\n--- {res['label']} ---")
    print(f"  population {res['n']:,}  mules {res['n_positive']} "
          f"({res['prevalence']*100:.2f}%)")
    print(f"  PR-AUC              {res['pr_auc']:.3f}")
    print(f"  Precision@{budget}        {res[f'precision@{budget}']:.3f}")
    print(f"  Recall@{budget}           {res[f'recall@{budget}']:.3f}")
    for d in ("blatant", "subtle"):
        if f"{d}_pr_auc" in res:
            print(f"  {d:8s} P@{budget} {res[f'{d}_precision@{budget}']:.3f}  "
                  f"R@{budget} {res[f'{d}_recall@{budget}']:.3f}  "
                  f"PR-AUC {res[f'{d}_pr_auc']:.3f}")
    print(f"  (ROC-AUC {res['roc_auc']:.3f} -- reported only to show how much "
          f"it flatters)")


def ablation_table(results, budget: int = 200) -> pd.DataFrame:
    """The headline table: what each feature layer actually buys."""
    rows = []
    for r in results:
        rows.append({
            "config": r["label"],
            "PR-AUC": round(r["pr_auc"], 3),
            f"P@{budget}": round(r[f"precision@{budget}"], 3),
            f"R@{budget}": round(r[f"recall@{budget}"], 3),
            "blatant R": round(r.get(f"blatant_recall@{budget}", float("nan")), 3),
            "subtle R": round(r.get(f"subtle_recall@{budget}", float("nan")), 3),
            "subtle P": round(r.get(f"subtle_precision@{budget}", float("nan")), 3),
        })
    return pd.DataFrame(rows)
