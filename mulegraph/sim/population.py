"""Population and node bookkeeping.

Internal accounts get ids 0..n-1. Everything outside the platform
(employers, landlords, merchants, retail customers, fraud victims, cash-out
points) gets an id in the external range, allocated on demand.

Channel codes are kept because real ledgers have them and they are a weak
but honest signal.
"""

import numpy as np
import pandas as pd

CH_UPI, CH_IMPS, CH_NEFT, CH_CARD, CH_CASH = 0, 1, 2, 3, 4
CHANNEL_NAMES = {0: "upi", 1: "imps", 2: "neft", 3: "card", 4: "cash"}

EXT_BASE = 10_000_000  # external ids start here


class NodeRegistry:
    """Allocates external counterparty ids and remembers what kind they are."""

    def __init__(self):
        self._next = EXT_BASE
        self.kinds: dict[int, str] = {}

    def new(self, kind: str) -> int:
        nid = self._next
        self._next += 1
        self.kinds[nid] = kind
        return nid

    def new_many(self, kind: str, n: int) -> np.ndarray:
        ids = np.arange(self._next, self._next + n, dtype=np.int64)
        self._next += n
        for i in ids:
            self.kinds[int(i)] = kind
        return ids

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "node_id": list(self.kinds.keys()),
            "kind": list(self.kinds.values()),
        })


class TxnBuffer:
    """Collects transactions as numpy chunks, concatenated once at the end.

    Appending 5M+ python tuples is what makes naive simulators unusable at
    scale; chunked arrays keep the whole run in a few hundred MB.
    """

    def __init__(self):
        self._ts, self._src, self._dst = [], [], []
        self._amt, self._ch = [], []

    def add(self, ts, src, dst, amount, channel):
        n = len(ts)
        if n == 0:
            return
        self._ts.append(np.asarray(ts, dtype=np.int32))
        self._src.append(np.broadcast_to(np.asarray(src, dtype=np.int64), (n,)).copy())
        self._dst.append(np.broadcast_to(np.asarray(dst, dtype=np.int64), (n,)).copy())
        self._amt.append(np.asarray(amount, dtype=np.float32))
        self._ch.append(np.broadcast_to(np.asarray(channel, dtype=np.int8), (n,)).copy())

    def to_frame(self) -> pd.DataFrame:
        if not self._ts:
            return pd.DataFrame(columns=["ts_min", "src", "dst", "amount", "channel"])
        df = pd.DataFrame({
            "ts_min": np.concatenate(self._ts),
            "src": np.concatenate(self._src),
            "dst": np.concatenate(self._dst),
            "amount": np.concatenate(self._amt),
            "channel": np.concatenate(self._ch),
        })
        df = df.sort_values("ts_min", kind="stable", ignore_index=True)
        df.insert(0, "txn_id", np.arange(len(df), dtype=np.int64))
        return df


def build_population(cfg, rng: np.random.Generator):
    """Assign personas, devices and IP prefixes to internal accounts."""
    n = cfg.n_accounts
    personas = list(cfg.persona_mix.keys())
    probs = np.array([cfg.persona_mix[p] for p in personas], dtype=float)
    probs = probs / probs.sum()
    persona = rng.choice(personas, size=n, p=probs)

    # One device per account by default; a few families share.
    device_id = np.arange(n, dtype=np.int64)
    n_share = int(cfg.normal_device_share_rate * n)
    if n_share >= 2:
        sharers = rng.choice(n, size=n_share, replace=False)
        # pair them up: second of each pair adopts the first's device
        for i in range(0, len(sharers) - 1, 2):
            device_id[sharers[i + 1]] = device_id[sharers[i]]

    # IP prefixes are coarse (an ISP block), so many unrelated people share one.
    # This is deliberate noise: shared IP alone must not be a giveaway.
    ip_prefix = rng.integers(0, max(50, n // 40), size=n)

    accounts = pd.DataFrame({
        "account_id": np.arange(n, dtype=np.int64),
        "persona": persona,
        "device_id": device_id,
        "ip_prefix": ip_prefix,
        "is_mule": np.zeros(n, dtype=bool),
        "ring_id": np.full(n, -1, dtype=np.int64),
        "mule_difficulty": np.array([""] * n, dtype=object),
        "activation_day": np.full(n, -1, dtype=np.int64),
        "life_event": np.array([""] * n, dtype=object),
        "life_event_day": np.full(n, -1, dtype=np.int64),
    })
    return accounts


def assign_life_events(cfg, accounts: pd.DataFrame, rng: np.random.Generator):
    """Life events land late so they break the account's own 8-week baseline.

    This is the deliberate cruelty of the dataset: a wedding produces dozens
    of one-time senders and a fast outflow, which is the same shape as a
    freshly activated mule.
    """
    n = len(accounts)
    n_events = int(cfg.life_event_rate * n)
    if n_events == 0:
        return accounts
    idx = rng.choice(n, size=n_events, replace=False)
    kinds = list(cfg.life_event_mix.keys())
    p = np.array([cfg.life_event_mix[k] for k in kinds], dtype=float)
    p = p / p.sum()
    chosen = rng.choice(kinds, size=n_events, p=p)
    # events occur in the last 3 weeks of the window
    day = rng.integers(cfg.n_days - 21, cfg.n_days - 2, size=n_events)
    accounts.loc[idx, "life_event"] = chosen
    accounts.loc[idx, "life_event_day"] = day
    return accounts
