"""Account-level feature extraction.

This module measures behaviour from the raw ledger. It never reads
SimConfig and never looks at labels. If you can compute a feature here
from a real bank extract, it belongs; if it needs the generator, it does
not.

Features are grouped so the ablation can turn layers on and off:

    GROUP_RAW      week-9 behaviour only
    GROUP_BASELINE + the account's own prior 8 weeks
    GROUP_GRAPH    + ring-level structure (added by graph/features.py)

That ordering is the experiment: it shows how much of subtle-mule recall
comes from each layer rather than asserting that the graph earns its place.
"""

from __future__ import annotations

import glob
import numpy as np
import pandas as pd

EXT_BASE = 10_000_000
DAY = 1440

GROUP_RAW = [
    "pass_through", "drain_median_h", "drain_p25_h", "fan_in", "fan_out",
    "n_credits", "n_debits", "credit_amount", "fan_ratio",
    "amount_round_frac", "night_frac", "credit_concentration",
    "same_day_out_frac", "median_credit", "credit_cv",
]

GROUP_BASELINE = [
    "throughput_jump", "fanin_jump", "new_beneficiary_frac",
    "sender_repeat_rate", "beneficiary_stability", "active_days_ratio",
    "days_since_first_activity", "baseline_weeks_active", "dormant_before",
]


def load_data(data_dir: str = "data"):
    def _load(base):
        for ext in (".parquet", ".csv.gz"):
            hits = glob.glob(f"{data_dir}/{base}{ext}")
            if hits:
                return (pd.read_parquet(hits[0]) if ext == ".parquet"
                        else pd.read_csv(hits[0]))
        raise FileNotFoundError(f"{data_dir}/{base}")

    txns = _load("transactions")
    accounts = _load("accounts")
    accounts["mule_difficulty"] = accounts["mule_difficulty"].fillna("")
    accounts["life_event"] = accounts["life_event"].fillna("")
    return txns, accounts


def _split_legs(df):
    """Split transactions into per-account credit and debit views."""
    cred = df[df["dst"] < EXT_BASE][["dst", "src", "amount", "ts_min", "day"]].copy()
    cred.columns = ["acct", "cp", "amount", "ts_min", "day"]
    deb = df[df["src"] < EXT_BASE][["src", "dst", "amount", "ts_min", "day"]].copy()
    deb.columns = ["acct", "cp", "amount", "ts_min", "day"]
    return cred, deb


def build_features(txns: pd.DataFrame, accounts: pd.DataFrame,
                   window_days: int = 7) -> pd.DataFrame:
    """One row per account: week-9 behaviour plus deviation from its own past.

    The temporal split is the point. Baseline comes from days before the
    scoring window and the label is evaluated on the window itself, so
    nothing from the future leaks backwards. A random train/test split on
    this data would be meaningless.
    """
    n_days = int(txns["day"].max()) + 1
    start = n_days - window_days

    win = txns[txns["day"] >= start]
    past = txns[txns["day"] < start]

    cw, dw = _split_legs(win)
    cp_, dp_ = _split_legs(past)

    # ---- window behaviour -------------------------------------------
    ci = cw.groupby("acct").agg(
        credit_amount=("amount", "sum"),
        n_credits=("amount", "size"),
        fan_in=("cp", "nunique"),
        median_credit=("amount", "median"),
        credit_std=("amount", "std"),
    )
    di = dw.groupby("acct").agg(
        debit_amount=("amount", "sum"),
        n_debits=("amount", "size"),
        fan_out=("cp", "nunique"),
    )
    f = ci.join(di, how="outer")
    f[["credit_amount", "debit_amount", "n_credits", "n_debits",
       "fan_in", "fan_out"]] = f[["credit_amount", "debit_amount", "n_credits",
                                  "n_debits", "fan_in", "fan_out"]].fillna(0.0)

    f["pass_through"] = (f["debit_amount"] / f["credit_amount"].clip(lower=1.0)).clip(0, 3)
    f["fan_ratio"] = f["fan_in"] / f["fan_out"].clip(lower=1)
    f["credit_cv"] = (f["credit_std"] / f["median_credit"].clip(lower=1)).fillna(0)

    # largest single sender's share -- a shopkeeper's take is spread across
    # many customers, a mule's inflow is often lumpier
    top = cw.groupby(["acct", "cp"])["amount"].sum().groupby("acct").max()
    f["credit_concentration"] = (top / f["credit_amount"].clip(lower=1)).fillna(0)

    # round amounts: victim transfers are chunkier than retail spend
    cw = cw.assign(is_round=(cw["amount"] % 500 == 0))
    f["amount_round_frac"] = cw.groupby("acct")["is_round"].mean().fillna(0)

    # activity outside business hours
    dw = dw.assign(hour=(dw["ts_min"] % DAY) // 60)
    f["night_frac"] = dw.assign(
        night=((dw["hour"] < 7) | (dw["hour"] >= 22))
    ).groupby("acct")["night"].mean().fillna(0)

    # ---- drain speed -------------------------------------------------
    c_sorted = cw[["acct", "ts_min"]].sort_values("ts_min")
    d_sorted = dw[["acct", "ts_min"]].sort_values("ts_min").rename(
        columns={"ts_min": "t_deb"})
    m = pd.merge_asof(c_sorted, d_sorted, left_on="ts_min", right_on="t_deb",
                      by="acct", direction="forward")
    m["gap_h"] = (m["t_deb"] - m["ts_min"]) / 60.0
    g = m.groupby("acct")["gap_h"]
    f["drain_median_h"] = g.median()
    f["drain_p25_h"] = g.quantile(0.25)
    f["same_day_out_frac"] = m.assign(
        same=(m["gap_h"] <= 24)).groupby("acct")["same"].mean()

    # ---- the account's own prior 8 weeks -----------------------------
    base_weeks = max(1.0, start / 7.0)
    pb = cp_.groupby("acct").agg(
        base_credit=("amount", "sum"),
        base_fan_in=("cp", "nunique"),
        base_days=("day", "nunique"),
        first_day=("day", "min"),
    )
    f = f.join(pb)
    f["base_credit"] = f["base_credit"].fillna(0.0)
    f["base_fan_in"] = f["base_fan_in"].fillna(0.0)
    f["base_days"] = f["base_days"].fillna(0.0)

    f["throughput_jump"] = f["credit_amount"] / (
        f["base_credit"] / base_weeks).clip(lower=500.0)
    f["fanin_jump"] = f["fan_in"] / (
        f["base_fan_in"] / base_weeks).clip(lower=0.5)
    f["baseline_weeks_active"] = f["base_days"] / 7.0
    f["days_since_first_activity"] = (n_days - f["first_day"].fillna(n_days))
    f["dormant_before"] = (f["base_days"] < 3).astype(int)
    f["active_days_ratio"] = win.groupby(
        win["src"].where(win["src"] < EXT_BASE, win["dst"])
    )["day"].nunique().reindex(f.index).fillna(0) / window_days

    # beneficiaries this week that were never paid before
    prev_ben = dp_.groupby("acct")["cp"].apply(set)
    now_ben = dw.groupby("acct")["cp"].apply(set)
    both = pd.DataFrame({"prev": prev_ben, "now": now_ben})
    both["prev"] = both["prev"].apply(lambda s: s if isinstance(s, set) else set())
    both["now"] = both["now"].apply(lambda s: s if isinstance(s, set) else set())
    f["new_beneficiary_frac"] = both.apply(
        lambda r: (len(r["now"] - r["prev"]) / max(1, len(r["now"])))
        if r["now"] else 0.0, axis=1)
    f["beneficiary_stability"] = both.apply(
        lambda r: (len(r["now"] & r["prev"]) / max(1, len(r["now"] | r["prev"])))
        if (r["now"] or r["prev"]) else 0.0, axis=1)

    # ---- sender repeat rate, over FULL history -----------------------
    # A student's parent sends once a month; a 7-day window can never show a
    # repeat. Repeat behaviour is a property of a relationship over time, so
    # it must be measured over the whole ledger.
    call_ = txns[txns["dst"] < EXT_BASE][["dst", "src"]]
    call_.columns = ["acct", "cp"]
    vc = call_.groupby(["acct", "cp"]).size().rename("k").reset_index()
    vc["rep_k"] = np.where(vc["k"] > 1, vc["k"], 0)
    agg = vc.groupby("acct").agg(rep_k=("rep_k", "sum"), tot_k=("k", "sum"))
    f["sender_repeat_rate"] = (agg["rep_k"] / agg["tot_k"].clip(lower=1))

    # ---- assemble ----------------------------------------------------
    out = accounts.set_index("account_id").join(f, how="left")
    for c in GROUP_RAW + GROUP_BASELINE:
        if c not in out.columns:
            out[c] = 0.0
    out[GROUP_RAW + GROUP_BASELINE] = out[GROUP_RAW + GROUP_BASELINE].fillna(0.0)
    out = out.replace([np.inf, -np.inf], 0.0)
    return out.reset_index()


def write_table(df: pd.DataFrame, base: str) -> str:
    """Parquet when pyarrow is around, gzipped CSV otherwise."""
    try:
        path = base + ".parquet"
        df.to_parquet(path, index=False)
        return path
    except Exception:
        path = base + ".csv.gz"
        df.to_csv(path, index=False, compression="gzip")
        return path


def load_features(data_dir: str = "data") -> pd.DataFrame:
    for ext in (".parquet", ".csv.gz"):
        hits = glob.glob(f"{data_dir}/features{ext}")
        if hits:
            return (pd.read_parquet(hits[0]) if ext == ".parquet"
                    else pd.read_csv(hits[0]))
    raise FileNotFoundError("run: python -m mulegraph.features.account")


if __name__ == "__main__":
    txns, accounts = load_data()
    feats = build_features(txns, accounts)
    print(f"features: {len(feats):,} accounts x "
          f"{len(GROUP_RAW) + len(GROUP_BASELINE)} features")
    print(f"mules: {int(feats['is_mule'].sum())} "
          f"(subtle {int((feats['mule_difficulty']=='subtle').sum())}, "
          f"blatant {int((feats['mule_difficulty']=='blatant').sum())})")
    print("wrote", write_table(feats, "data/features"))
