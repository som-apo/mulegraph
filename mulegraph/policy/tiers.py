"""Tiered response policy and reason codes.

The model ranks. This module decides what happens. Keeping those two
things separate is the single most important design decision in the
project, because it changes what a false positive costs.

WHY NOT JUST FREEZE
-------------------
A false positive here is not an abstract unit of analyst time. It is a
person locked out of their own money. And it is a predictable set of
people: students, shopkeepers sweeping the day's UPI take to a supplier,
freelancers with irregular multi-payer income, someone who just moved
city, a family sharing one phone, a wedding, a medical emergency. Not
salaried customers with stable inflows, who never trip the model at all.
The cost of our errors falls hardest on the people least able to absorb
it, and there has been real public criticism in India of innocent accounts
frozen for weeks over money that merely passed through.

WHY THE LADDER WINS
-------------------
1. Most errors cost nothing. Thousands of false positives that only
   trigger silent monitoring harm nobody. In a freeze-or-nothing system
   those same people lose access to their money.
2. We can afford high recall, because the action at a low threshold is
   cheap. A binary system structurally cannot make that trade.
3. Every alert carries reason codes. Operations cannot act on a bare
   score, and a customer deserves an explanation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIERS = [
    ("critical", 0.9995, "outbound hold + escalate", 2000.0),
    ("high",     0.995,  "manual review",            200.0),
    ("medium",   0.98,   "transfer-limit cap",        50.0),
    ("low",      0.90,   "silent monitoring",          0.0),
    ("none",     0.0,    "no action",                  0.0),
]

# Thresholds for turning a feature value into a human sentence. These are
# explanation thresholds, not detection thresholds -- the model does the
# detecting, these only describe what it saw.
REASONS = [
    ("throughput_jump", 5.0, "{v:.0f}x throughput jump vs own 8-week baseline"),
    ("pass_through", 0.85, "{v:.0%} of credits forwarded on"),
    ("drain_median_h", 12.0, "median {v:.1f}h from credit to outflow", "below"),
    ("new_beneficiary_frac", 0.7, "{v:.0%} of beneficiaries are new"),
    ("fan_in", 10, "{v:.0f} distinct senders this week"),
    ("sender_repeat_rate", 0.4, "only {v:.0%} of inflow from repeat senders", "below"),
    ("device_share_count", 2, "device shared with {v0:.0f} other accounts"),
    ("consolidator_score", 3, "receives from {v:.0f} on-platform accounts"),
    ("cluster_size", 8, "sits in a cluster of {v:.0f} connected accounts"),
    ("amount_round_frac", 0.5, "{v:.0%} of credits are round amounts"),
]


def assign_tiers(scores) -> pd.Series:
    """Map scores to risk bands by quantile.

    Quantiles rather than fixed score cut-offs, because a probability from
    one training run is not comparable to one from the next, but "the top
    0.5% of today's queue" always means the same thing to an operations
    team sizing its day.
    """
    s = pd.Series(np.asarray(scores, dtype=float))
    q = s.rank(pct=True)
    tier = pd.Series("none", index=s.index, dtype=object)
    for name, floor, _, _ in sorted(TIERS, key=lambda t: t[1]):
        tier[q >= floor] = name
    return tier


def reason_codes(row: pd.Series, max_codes: int = 4) -> str:
    """Plain-language explanation of why this account surfaced."""
    out = []
    for spec in REASONS:
        col, thresh, template = spec[0], spec[1], spec[2]
        direction = spec[3] if len(spec) > 3 else "above"
        if col not in row or pd.isna(row[col]):
            continue
        v = float(row[col])
        hit = (v <= thresh) if direction == "below" else (v >= thresh)
        if hit:
            out.append(template.format(v=v, v0=max(0.0, v - 1)))
        if len(out) >= max_codes:
            break
    return "; ".join(out) if out else "combination of weak signals"


def build_queue(feats: pd.DataFrame, score_col: str = "score",
                budget: int = 200) -> pd.DataFrame:
    """The actual deliverable: a ranked daily review queue."""
    f = feats.copy()
    f["tier"] = assign_tiers(f[score_col].to_numpy())
    action = {name: act for name, _, act, _ in TIERS}
    f["action"] = f["tier"].map(action)

    q = f.nlargest(budget, score_col).copy()
    q["reason_codes"] = q.apply(reason_codes, axis=1)
    q["rank"] = np.arange(1, len(q) + 1)

    cols = ["rank", "account_id", score_col, "tier", "action", "reason_codes"]
    if "cluster_id" in q.columns:
        cols.insert(-1, "cluster_id")
    return q[cols].reset_index(drop=True)


def ring_view(feats: pd.DataFrame, score_col: str = "score",
              min_size: int = 4, top_n: int = 10) -> pd.DataFrame:
    """Cluster-level output.

    "These 14 accounts are one operation" is a far more useful thing to
    hand an analyst than fourteen separate alerts, and it is the difference
    between blocking an operation and playing whack-a-mole while the
    operator activates the next account of thirty.
    """
    if "cluster_id" not in feats.columns:
        return pd.DataFrame()
    g = feats[feats["cluster_id"] >= 0].groupby("cluster_id")
    r = g.agg(accounts=(score_col, "size"),
              mean_score=(score_col, "mean"),
              max_score=(score_col, "max"),
              median_pass_through=("pass_through", "median"),
              median_fan_in=("fan_in", "median"))
    r = r[r["accounts"] >= min_size]
    return r.nlargest(top_n, "mean_score").reset_index()


def policy_cost(feats: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """What the tiered ladder costs when it is wrong, band by band.

    The headline of the whole project lives in the error-cost column: the
    overwhelming majority of our mistakes cost the customer nothing,
    because the action at that confidence level is to watch, not to act.
    """
    f = feats.copy()
    f["tier"] = assign_tiers(f[score_col].to_numpy())
    cost = {name: c for name, _, _, c in TIERS}
    act = {name: a for name, _, a, _ in TIERS}

    rows = []
    for name in ["critical", "high", "medium", "low", "none"]:
        sl = f[f["tier"] == name]
        if sl.empty:
            continue
        fp = int((~sl["is_mule"].astype(bool)).sum())
        rows.append({
            "tier": name,
            "action": act[name],
            "accounts": len(sl),
            "mules": int(sl["is_mule"].sum()),
            "false_positives": fp,
            "cost_per_error": cost[name],
            "total_error_cost": fp * cost[name],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import glob
    from ..features.account import write_table

    hits = glob.glob("data/scored.parquet") or glob.glob("data/scored.csv.gz")
    if not hits:
        raise SystemExit("run scripts/run_pipeline.py first")
    f = (pd.read_parquet(hits[0]) if hits[0].endswith("parquet")
         else pd.read_csv(hits[0]))

    q = build_queue(f, budget=200)
    print("=== ranked review queue (top 10 of 200) ===")
    print(q.head(10).to_string(index=False))

    print("\n=== ring view: highest-scoring clusters ===")
    print(ring_view(f).to_string(index=False))

    print("\n=== what our errors actually cost ===")
    pc = policy_cost(f)
    print(pc.to_string(index=False))
    zero = pc[pc["cost_per_error"] == 0]["false_positives"].sum()
    tot = pc["false_positives"].sum()
    print(f"\n{zero:,} of {tot:,} false positives ({zero/max(1,tot)*100:.1f}%) "
          f"cost the customer nothing.")

    print("\nwrote", write_table(q, "data/review_queue"))
