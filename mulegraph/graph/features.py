"""Money-flow graph and ring-level structure.

The account-level model asks "does this account behave like a pipe?".
This module asks a different question: "is this account part of an
operation?" -- and that question survives an adversary who has learned to
make each individual account look ordinary.

THE ARGUMENT
------------
To look normal on the account-level signals, a mule must slow down and
hold money back, which cuts its throughput. To keep the operation's total
throughput, the operator must recruit more accounts. More accounts means a
denser, more connected ring. So evading the account model pushes you into
the graph model.

In this repo that is not a rhetorical claim: the simulator derives ring
size by dividing a fixed rupee target by per-account throughput, so subtle
rings really are ~7x larger than blatant ones.

WHAT WE ARE CAREFUL ABOUT
-------------------------
Legitimate accounts form dense communities too -- shared employers, small
businesses paying on-platform staff, suppliers who bank here. If only
mules clustered, community detection would win trivially and this whole
layer would be a circular artefact of the generator rather than a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

EXT_BASE = 10_000_000
DAY = 1440

GROUP_GRAPH = [
    "device_share_count", "ip_share_count",
    "internal_out_frac", "internal_in_frac",
    "cluster_size", "cluster_pass_through", "cluster_mule_density_proxy",
    "cluster_mean_fanin", "cluster_new_ben_frac", "cluster_burst_sync",
    "counterparty_reuse_across_accounts", "two_hop_internal_reach",
    "consolidator_score",
]


def build_flow_graph(txns: pd.DataFrame, window_days: int = 21) -> nx.Graph:
    """Undirected weighted graph over accounts that moved money between
    each other.

    Undirected because community detection needs it, and because for
    "are these accounts one operation?" direction matters less than
    connection. Direction is preserved separately in the degree features.
    """
    n_days = int(txns["day"].max()) + 1
    w = txns[txns["day"] >= n_days - window_days]

    internal = w[(w["src"] < EXT_BASE) & (w["dst"] < EXT_BASE)]
    agg = internal.groupby(["src", "dst"])["amount"].agg(["sum", "size"]).reset_index()

    G = nx.Graph()
    for s, d, amt, cnt in agg[["src", "dst", "sum", "size"]].itertuples(index=False):
        if G.has_edge(s, d):
            G[s][d]["weight"] += float(amt)
            G[s][d]["count"] += int(cnt)
        else:
            G.add_edge(int(s), int(d), weight=float(amt), count=int(cnt))
    return G


def shared_counterparty_edges(txns: pd.DataFrame, min_shared: int = 2,
                              max_hub_degree: int = 40) -> pd.DataFrame:
    """Accounts linked by paying or receiving from the same outside party.

    Hubs are excluded. Ten thousand people shop at the same supermarket;
    that tells us nothing. Two accounts sharing three obscure beneficiaries
    tells us a great deal. Without the hub cap this feature would connect
    the entire population through a handful of merchants.
    """
    ext = txns[(txns["src"] >= EXT_BASE) | (txns["dst"] >= EXT_BASE)].copy()
    ext["acct"] = np.where(ext["src"] < EXT_BASE, ext["src"], ext["dst"])
    ext["cp"] = np.where(ext["src"] < EXT_BASE, ext["dst"], ext["src"])

    pairs = ext[["acct", "cp"]].drop_duplicates()
    deg = pairs.groupby("cp").size()
    keep = deg[(deg <= max_hub_degree) & (deg > 1)].index
    pairs = pairs[pairs["cp"].isin(keep)]

    m = pairs.merge(pairs, on="cp")
    m = m[m["acct_x"] < m["acct_y"]]
    out = m.groupby(["acct_x", "acct_y"]).size().rename("shared").reset_index()
    return out[out["shared"] >= min_shared]


def detect_communities(G: nx.Graph, seed: int = 0) -> dict:
    """Louvain if available, connected components as a fallback."""
    try:
        import community as community_louvain
        return community_louvain.best_partition(G, random_state=seed)
    except Exception:
        try:
            comms = nx.community.louvain_communities(G, seed=seed)
        except Exception:
            comms = nx.connected_components(G)
        return {n: i for i, c in enumerate(comms) for n in c}


def build_graph_features(txns: pd.DataFrame, accounts: pd.DataFrame,
                         feats: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Attach graph and cluster features to the account feature table."""
    n_days = int(txns["day"].max()) + 1
    start = n_days - window_days
    win = txns[txns["day"] >= start]

    idx = feats.set_index("account_id")
    out = pd.DataFrame(index=idx.index)

    # ---- shared infrastructure ---------------------------------------
    # Sparse by construction in the simulator: sloppy rings share devices,
    # careful ones mostly do not, and real families share phones. This is a
    # supporting signal, never a decisive one.
    dev = accounts.groupby("device_id")["account_id"].size()
    ipc = accounts.groupby("ip_prefix")["account_id"].size()
    a = accounts.set_index("account_id")
    out["device_share_count"] = a["device_id"].map(dev).reindex(out.index).fillna(1)
    out["ip_share_count"] = a["ip_prefix"].map(ipc).reindex(out.index).fillna(1)

    # ---- how much money stays on-platform ----------------------------
    deb = win[win["src"] < EXT_BASE]
    cre = win[win["dst"] < EXT_BASE]
    do = deb.assign(internal=(deb["dst"] < EXT_BASE)).groupby("src")["internal"].mean()
    di = cre.assign(internal=(cre["src"] < EXT_BASE)).groupby("dst")["internal"].mean()
    out["internal_out_frac"] = do.reindex(out.index).fillna(0.0)
    out["internal_in_frac"] = di.reindex(out.index).fillna(0.0)

    # A consolidator receives from many on-platform accounts and forwards
    # almost everything straight out. That shape is rare in ordinary life.
    in_int = cre[cre["src"] < EXT_BASE].groupby("dst")["src"].nunique()
    out["consolidator_score"] = (
        in_int.reindex(out.index).fillna(0) * out["internal_in_frac"]
    )

    # ---- communities on the money-flow graph -------------------------
    G = build_flow_graph(txns, window_days=21)
    sc = shared_counterparty_edges(txns)
    for x, y, w in sc[["acct_x", "acct_y", "shared"]].itertuples(index=False):
        if G.has_edge(x, y):
            G[x][y]["weight"] += float(w) * 1000.0
        else:
            G.add_edge(int(x), int(y), weight=float(w) * 1000.0, count=0)

    part = detect_communities(G)
    out["cluster_id"] = pd.Series(part).reindex(out.index).fillna(-1).astype(int)

    # ---- cluster aggregates ------------------------------------------
    # Score the group, not just the account. Output is "these 14 accounts
    # are one operation", not "freeze this account".
    base = idx[["pass_through", "fan_in", "new_beneficiary_frac",
                "n_credits", "credit_amount"]].copy()
    base["cluster_id"] = out["cluster_id"]
    base["active_day_span"] = idx["active_days_ratio"]

    grp = base[base["cluster_id"] >= 0].groupby("cluster_id")
    cl = grp.agg(
        cluster_size=("pass_through", "size"),
        cluster_pass_through=("pass_through", "median"),
        cluster_mean_fanin=("fan_in", "median"),
        cluster_new_ben_frac=("new_beneficiary_frac", "median"),
        cluster_burst_sync=("active_day_span", "std"),
    )
    # a proxy for "how pipe-like is this whole cluster", computed from
    # behaviour only -- never from labels
    cl["cluster_mule_density_proxy"] = (
        cl["cluster_pass_through"].clip(0, 1.5)
        * (cl["cluster_size"] / max(1, cl["cluster_size"].max()))
    )
    cl["cluster_burst_sync"] = cl["cluster_burst_sync"].fillna(0.0)

    for c in ["cluster_size", "cluster_pass_through", "cluster_mean_fanin",
              "cluster_new_ben_frac", "cluster_burst_sync",
              "cluster_mule_density_proxy"]:
        out[c] = out["cluster_id"].map(cl[c]).fillna(0.0)

    # ---- neighbourhood -----------------------------------------------
    reuse = sc.groupby("acct_x")["shared"].sum().add(
        sc.groupby("acct_y")["shared"].sum(), fill_value=0)
    out["counterparty_reuse_across_accounts"] = reuse.reindex(out.index).fillna(0.0)

    two_hop = {}
    for n in list(G.nodes):
        if n >= EXT_BASE:
            continue
        nb = list(G.neighbors(n))
        two = set()
        for m in nb[:50]:
            two |= set(G.neighbors(m))
        two_hop[n] = len(two - {n})
    out["two_hop_internal_reach"] = pd.Series(two_hop).reindex(out.index).fillna(0.0)

    merged = feats.merge(out.reset_index(), on="account_id", how="left")
    merged[GROUP_GRAPH] = merged[GROUP_GRAPH].fillna(0.0)
    return merged.replace([np.inf, -np.inf], 0.0)


if __name__ == "__main__":
    from ..features.account import load_data, load_features, write_table

    txns, accounts = load_data()
    feats = load_features()
    g = build_graph_features(txns, accounts, feats)
    print(f"graph features attached: {len(GROUP_GRAPH)} columns")
    print(f"clusters found: {g['cluster_id'].nunique():,}")
    big = g[g["cluster_size"] >= 5]
    print(f"accounts in clusters of 5+: {len(big):,} "
          f"(of which mules: {int(big['is_mule'].sum())})")
    print("wrote", write_table(g, "data/features_graph"))
