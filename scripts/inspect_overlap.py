"""GATE 1: does the simulator produce a genuinely hard problem?

This is diagnostic code, not the real feature module. It computes rough
versions of the five signals and asks one question: do subtle mules sit
inside the same region of feature space as shopkeepers, students and
wedding accounts?

If mules separate cleanly on any single feature, the simulator is leaking
its answer key and every downstream number is worthless. We want to SEE
the overlap before we train anything.
"""

import glob
import numpy as np
import pandas as pd

EXT_BASE = 10_000_000
DAY = 1440


def _load(base):
    for ext in (".parquet", ".csv.gz"):
        hits = glob.glob(base + ext)
        if hits:
            return (pd.read_parquet(hits[0]) if ext == ".parquet"
                    else pd.read_csv(hits[0]))
    raise FileNotFoundError(base)


def account_signals(txns, accounts, window_days=7):
    """Rough five-signal snapshot over the scoring window."""
    n_days = int(txns["day"].max()) + 1
    start = n_days - window_days
    w = txns[txns["day"] >= start]

    cred = w[w["dst"] < EXT_BASE][["dst", "src", "amount", "ts_min"]]
    cred.columns = ["acct", "cp", "amount", "ts_min"]
    deb = w[w["src"] < EXT_BASE][["src", "dst", "amount", "ts_min"]]
    deb.columns = ["acct", "cp", "amount", "ts_min"]

    cred = cred.assign(is_round=(cred["amount"] % 500 == 0))
    ci = cred.groupby("acct").agg(credits=("amount", "sum"),
                                  amount_round_frac=("is_round", "mean"),
                                  n_credits=("amount", "size"),
                                  fan_in=("cp", "nunique"))
    di = deb.groupby("acct").agg(debits=("amount", "sum"),
                                 fan_out=("cp", "nunique"))
    f = ci.join(di, how="outer").fillna(0.0)
    f["pass_through"] = f["debits"] / f["credits"].clip(lower=1.0)
    f["pass_through"] = f["pass_through"].clip(0, 2.0)

    # How often does the same sender come back? Measured over FULL history,
    # not the scoring window -- a student's parent sends once a month, so a
    # 7-day window can never show a repeat and the feature reads as 0 for
    # everyone. Repeat behaviour is a property of a relationship over time.
    cred_all = txns[txns["dst"] < EXT_BASE][["dst", "src"]]
    cred_all.columns = ["acct", "cp"]
    vc = cred_all.groupby(["acct", "cp"]).size().rename("k").reset_index()
    vc["rep_k"] = np.where(vc["k"] > 1, vc["k"], 0)
    g = vc.groupby("acct").agg(rep_k=("rep_k", "sum"), tot_k=("k", "sum"))
    rep = (g["rep_k"] / g["tot_k"].clip(lower=1)).rename("sender_repeat_rate")
    f = f.join(rep).fillna({"sender_repeat_rate": 0.0})

    # time from a credit to the next debit on the same account
    c = cred[["acct", "ts_min"]].sort_values("ts_min")
    d = deb[["acct", "ts_min"]].sort_values("ts_min").rename(columns={"ts_min": "t_deb"})
    m = pd.merge_asof(c, d, left_on="ts_min", right_on="t_deb", by="acct",
                      direction="forward")
    m["gap_h"] = (m["t_deb"] - m["ts_min"]) / 60.0
    drain = m.groupby("acct")["gap_h"].median().rename("drain_median_h")
    f = f.join(drain)

    # week 9 volume vs the account's own prior weekly average
    base = txns[txns["day"] < start]
    bc = base[base["dst"] < EXT_BASE].groupby("dst")["amount"].sum()
    weeks = max(1, start / 7.0)
    baseline_weekly = (bc / weeks).rename("baseline_weekly")
    f = f.join(baseline_weekly)
    f["throughput_jump"] = f["credits"] / f["baseline_weekly"].fillna(0).clip(lower=500.0)

    out = accounts.set_index("account_id").join(f, how="left")
    return out.fillna({"pass_through": 0, "fan_in": 0, "fan_out": 0,
                       "sender_repeat_rate": 0, "throughput_jump": 0,
                       "credits": 0, "n_credits": 0, "amount_round_frac": 0})


def report(sig):
    sig = sig.copy()
    sig["group"] = np.where(sig["is_mule"],
                            "MULE:" + sig["mule_difficulty"].astype(str),
                            np.where(sig["life_event"].astype(str) == "",
                                     sig["persona"], "event:" + sig["life_event"].astype(str)))

    # amount_round_frac is here because an earlier version leaked through it:
    # only mule transfers were rounded, so this single feature scored PR-AUC
    # 0.997. Any feature the model can see belongs in this gate.
    cols = ["pass_through", "drain_median_h", "fan_in", "fan_out",
            "sender_repeat_rate", "throughput_jump", "amount_round_frac"]
    act = sig[sig["n_credits"] > 0]

    print("\n=== median signal values by group (scoring window) ===")
    tab = act.groupby("group")[cols].median().round(2)
    tab["n"] = act.groupby("group").size()
    order = [g for g in ["MULE:blatant", "MULE:subtle", "shopkeeper",
                         "micro_merchant", "collector", "student",
                         "event:new_business", "event:wedding", "event:medical", "small_business",
                         "freelancer", "salaried", "low_activity",
                         "event:job_change", "event:city_move"] if g in tab.index]
    print(tab.loc[order].to_string())

    print("\n=== single-feature separability (AUC of one feature alone) ===")
    print("if any of these is near 1.00, the simulator is leaking its answer key")
    from sklearn.metrics import roc_auc_score
    y = act["is_mule"].astype(int).to_numpy()
    for c in cols:
        v = act[c].fillna(act[c].median()).to_numpy()
        try:
            a = roc_auc_score(y, v)
            print(f"  {c:22s} AUC {max(a, 1-a):.3f}")
        except Exception as e:
            print(f"  {c:22s} n/a ({e})")

    print("\n=== overlap: subtle mules vs hard negatives ===")
    sub = act[act["group"] == "MULE:subtle"]
    hard = act[act["group"].isin(["shopkeeper", "micro_merchant", "collector", "student",
                                  "event:wedding", "small_business"])]
    for c in ["pass_through", "drain_median_h", "fan_in"]:
        if sub.empty or hard.empty:
            continue
        lo, hi = np.nanpercentile(sub[c], [10, 90])
        inside = float(((hard[c] >= lo) & (hard[c] <= hi)).mean())
        print(f"  {c:22s} {inside*100:5.1f}% of hard negatives fall inside "
              f"the subtle-mule 10-90 band [{lo:.2f}, {hi:.2f}]")

    print("\n=== the squeeze ===")
    rings = _load("data/rings")
    if "difficulty" in rings:
        print(rings.groupby("difficulty")[["size", "per_account_throughput"]]
              .mean().round(0).to_string())


if __name__ == "__main__":
    txns = _load("data/transactions")
    accounts = _load("data/accounts")
    accounts["mule_difficulty"] = accounts["mule_difficulty"].fillna("")
    accounts["life_event"] = accounts["life_event"].fillna("")
    sig = account_signals(txns, accounts)
    report(sig)
