"""End-to-end pipeline.

    python scripts/run_pipeline.py --accounts 100000

Runs the whole thing and prints the ablation table that is the headline
result of the project.
"""

from __future__ import annotations

import argparse
import time
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from mulegraph.config import SimConfig
from mulegraph.sim.generate import simulate
from mulegraph.features.account import (
    load_data, build_features, write_table, GROUP_RAW, GROUP_BASELINE,
)
from mulegraph.graph.features import build_graph_features, GROUP_GRAPH
from mulegraph.models.account import cv_scores
from mulegraph.eval.metrics import evaluate, print_report, ablation_table, cost_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", type=int, default=100_000)
    ap.add_argument("--mule-rate", type=float, default=0.005)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-sim", action="store_true")
    args = ap.parse_args()

    t0 = time.time()

    if not args.skip_sim:
        print("=" * 68)
        print("STAGE 1  simulate")
        print("=" * 68)
        simulate(SimConfig(n_accounts=args.accounts, mule_rate=args.mule_rate,
                           seed=args.seed))

    print("\n" + "=" * 68)
    print("STAGE 2  account features")
    print("=" * 68)
    txns, accounts = load_data()
    feats = build_features(txns, accounts)
    print(f"  {len(feats):,} accounts x "
          f"{len(GROUP_RAW)+len(GROUP_BASELINE)} account features "
          f"({time.time()-t0:.0f}s)")

    print("\n" + "=" * 68)
    print("STAGE 3  graph features")
    print("=" * 68)
    feats = build_graph_features(txns, accounts, feats)
    print(f"  + {len(GROUP_GRAPH)} graph features, "
          f"{feats['cluster_id'].nunique():,} clusters "
          f"({time.time()-t0:.0f}s)")
    print("  wrote", write_table(feats, "data/features_graph"))

    print("\n" + "=" * 68)
    print("STAGE 4  ablation")
    print("=" * 68)
    y = feats["is_mule"].astype(int).to_numpy()
    d = feats["mule_difficulty"].astype(str).to_numpy()
    budget = args.budget
    print(f"  population {len(feats):,}  mules {y.sum()} "
          f"({y.mean()*100:.2f}%)  alert budget {budget}/day\n")

    results, scores = [], {}
    for label, cols in [
        ("A: raw behaviour", GROUP_RAW),
        ("B: + self-baseline", GROUP_RAW + GROUP_BASELINE),
        ("C: + graph", GROUP_RAW + GROUP_BASELINE + GROUP_GRAPH),
    ]:
        cols = [c for c in cols if c in feats.columns]
        oof = cv_scores(feats[cols], y)
        r = evaluate(y, oof, d, budget=budget, label=label)
        results.append(r)
        scores[label] = oof
        print_report(r, budget=budget)

    print("\n" + "=" * 68)
    print("ABLATION  -- what each layer actually buys")
    print("=" * 68)
    print(ablation_table(results, budget=budget).to_string(index=False))

    best = scores["C: + graph"]
    feats["score"] = best

    print("\n" + "=" * 68)
    print("COST CURVE  -- value prevented vs users disrupted")
    print("=" * 68)
    print(cost_curve(y, best).to_string(index=False))

    write_table(feats, "data/scored")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
