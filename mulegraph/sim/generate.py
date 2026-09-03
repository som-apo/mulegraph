from __future__ import annotations
"""Simulation orchestrator.

Order matters:
  1. build the population and give everyone an ordinary life
  2. recruit some of those accounts into rings
  3. TRUNCATE the recruit's ordinary life at their activation day
  4. layer ring behaviour on top

Step 3 is what makes the changepoint real. A rented account genuinely
stops being a student account and starts being a pipe.
"""

import os
import time

import numpy as np
import pandas as pd

from ..config import SimConfig
from . import personas as P
from . import rings as R
from .population import (
    NodeRegistry, TxnBuffer, build_population, assign_life_events, CHANNEL_NAMES,
)


def _write(df: pd.DataFrame, base: str) -> str:
    """Parquet when pyarrow is around, gzipped CSV otherwise."""
    try:
        path = base + ".parquet"
        df.to_parquet(path, index=False)
        return path
    except Exception:
        path = base + ".csv.gz"
        df.to_csv(path, index=False, compression="gzip")
        return path


def simulate(cfg: SimConfig | None = None, verbose: bool = True):
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()

    reg = NodeRegistry()
    buf = TxnBuffer()

    accounts = build_population(cfg, rng)
    accounts = assign_life_events(cfg, accounts, rng)

    # shared employer hubs give the graph legitimate high-degree nodes
    employer_hubs = reg.new_many("employer", max(8, cfg.n_accounts // 200))
    salaried_pool = accounts.loc[accounts["persona"] == "salaried", "account_id"].to_numpy()

    if verbose:
        print(f"[sim] population {cfg.n_accounts:,} accounts over {cfg.n_days} days")

    # ---- 1. everyone gets an ordinary life --------------------------
    for aid, persona in zip(accounts["account_id"].to_numpy(),
                            accounts["persona"].to_numpy()):
        gen = P.GENERATORS[persona]
        if persona == "small_business":
            gen(cfg, rng, reg, buf, int(aid), employer_hubs, internal_pool=salaried_pool)
        else:
            gen(cfg, rng, reg, buf, int(aid), employer_hubs)
        P.sprinkle_stranger_credits(cfg, rng, reg, buf, int(aid))

    ev = accounts[accounts["life_event"] != ""]
    for aid, kind, day in zip(ev["account_id"], ev["life_event"], ev["life_event_day"]):
        P.apply_life_event(cfg, rng, reg, buf, int(aid), kind, int(day))

    normal_df = buf.to_frame()
    if verbose:
        print(f"[sim] normal activity: {len(normal_df):,} txns  ({time.time()-t0:.1f}s)")

    # ---- 2. recruit and plan ----------------------------------------
    plan = R.plan_rings(cfg, rng)
    plan = R.recruit(cfg, accounts, plan, rng)

    ring_buf = TxnBuffer()
    accounts, plan = R.generate_rings(cfg, rng, reg, ring_buf, accounts, plan)
    ring_df = ring_buf.to_frame()

    # ---- 3. truncate the recruits' ordinary life --------------------
    act = accounts.loc[accounts["is_mule"], ["account_id", "activation_day"]]
    act = act[act["activation_day"] >= 0]
    cutoff = dict(zip(act["account_id"].astype(int), act["activation_day"].astype(int)))

    if cutoff:
        cut_ser = pd.Series(cutoff, dtype="int64")
        src_cut = normal_df["src"].map(cut_ser)
        dst_cut = normal_df["dst"].map(cut_ser)
        acct_cut = src_cut.fillna(dst_cut)
        day = normal_df["ts_min"] // 1440
        # The person still exists. Their salary still lands, their family
        # still sends money -- they rented the account, they did not vanish.
        # So we truncate their OUTGOING life (the spending pattern changes
        # because the pipe now dictates it) and leave incoming legitimate
        # credits alone. This also stops the repeat-sender feature from
        # becoming a perfect mule detector.
        is_outflow = src_cut.notna()
        drop = is_outflow & (day >= src_cut)
        keep_tail = drop & (np.random.default_rng(cfg.seed + 1).random(len(normal_df)) < 0.40)
        normal_df = normal_df[~(drop & ~keep_tail)].copy()

    # ---- 4. combine --------------------------------------------------
    txns = pd.concat([normal_df, ring_df], ignore_index=True)
    txns = txns.sort_values("ts_min", kind="stable", ignore_index=True)
    txns["txn_id"] = np.arange(len(txns), dtype=np.int64)
    txns["channel_name"] = txns["channel"].map(CHANNEL_NAMES)
    txns["day"] = (txns["ts_min"] // 1440).astype(np.int16)

    ring_df_meta = pd.DataFrame([{k: v for k, v in r.items() if k != "members"}
                                 for r in plan])

    os.makedirs(cfg.out_dir, exist_ok=True)
    p1 = _write(txns, os.path.join(cfg.out_dir, "transactions"))
    p2 = _write(accounts, os.path.join(cfg.out_dir, "accounts"))
    p3 = _write(reg.to_frame(), os.path.join(cfg.out_dir, "external_nodes"))
    p4 = _write(ring_df_meta, os.path.join(cfg.out_dir, "rings"))

    if verbose:
        n_mule = int(accounts["is_mule"].sum())
        sub = int((accounts["mule_difficulty"] == "subtle").sum())
        bla = int((accounts["mule_difficulty"] == "blatant").sum())
        print(f"[sim] rings: {len(plan)}  mules: {n_mule} "
              f"({n_mule/len(accounts)*100:.2f}%)  subtle {sub} / blatant {bla}")
        print(f"[sim] transactions: {len(txns):,}")
        print(f"[sim] wrote {p1}, {p2}, {p3}, {p4}  ({time.time()-t0:.1f}s)")

    return txns, accounts, plan


if __name__ == "__main__":
    simulate()
