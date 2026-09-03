"""Mule ring generation.

A mule account is not a fake account -- it is a real person's account that
has been rented. So rings are built by *recruiting* accounts that already
exist in the population, truncating their ordinary life at the activation
day and layering ring behaviour on top. The changepoint that the
self-baseline features detect therefore emerges naturally instead of being
painted on.

THE SQUEEZE
-----------
Each ring is given a throughput target in rupees. Ring size is derived:

    ring_size = ceil(ring_throughput / per_account_throughput)

A subtle mule holds a bigger cut and drains slowly, so it moves far less
money per account. To hit the same target the operator must recruit ~6x
more accounts, which makes the ring denser and far more visible to
community detection. Evading the account-level model pushes you into the
graph-level model -- and here that is arithmetic, not rhetoric.
"""

import numpy as np

from .population import CH_UPI, CH_IMPS, CH_NEFT

DAY = 1440


def _round_ish(rng, amounts):
    """Victim transfers are chunkier and rounder than retail spend."""
    out = amounts.copy()
    mask = rng.random(out.size) < 0.55
    out[mask] = np.round(out[mask] / 500.0) * 500.0
    return np.maximum(out, 500.0)


def plan_rings(cfg, rng):
    """Decide how many rings, of what difficulty, and how big each must be."""
    blat_lo, blat_hi = cfg.blatant_account_throughput
    sub_lo, sub_hi = cfg.subtle_account_throughput
    thr_mid = 0.5 * (cfg.ring_throughput_min + cfg.ring_throughput_max)

    blat_size = max(3, int(round(thr_mid / (0.5 * (blat_lo + blat_hi)))))
    sub_size = max(3, int(round(thr_mid / (0.5 * (sub_lo + sub_hi)))))

    f = cfg.subtle_ring_fraction
    avg_size = f * sub_size + (1 - f) * blat_size
    n_mules_target = cfg.n_accounts * cfg.mule_rate
    n_rings = max(3, int(round(n_mules_target / avg_size)))

    # Stratify rather than sample: at small n_rings a random draw can wipe
    # out one class entirely and silently break the blatant/subtle split.
    n_subtle = max(1, int(round(n_rings * f)))
    labels = ["subtle"] * n_subtle + ["blatant"] * max(1, n_rings - n_subtle)
    rng.shuffle(labels)

    rings = []
    for r, difficulty in enumerate(labels):
        throughput = rng.uniform(cfg.ring_throughput_min, cfg.ring_throughput_max)
        per_acct = (rng.uniform(*cfg.subtle_account_throughput) if difficulty == "subtle"
                    else rng.uniform(*cfg.blatant_account_throughput))
        size = max(3, int(np.ceil(throughput / per_acct)))
        # rings start working somewhere in the second half of the window
        start = int(rng.integers(cfg.baseline_weeks * 7 - 21, cfg.n_days - 10))
        rings.append({
            "ring_id": r,
            "difficulty": difficulty,
            "throughput": throughput,
            "per_account_throughput": per_acct,
            "size": size,
            "start_day": max(7, start),
        })
    return rings


def recruit(cfg, accounts, rings, rng):
    """Pick which existing accounts get rented, biased toward the personas
    that real recruiters target."""
    weights = np.array([cfg.recruit_weights.get(p, 0.01) for p in accounts["persona"]],
                       dtype=float)
    weights = weights / weights.sum()
    total = sum(r["size"] for r in rings)
    total = min(total, len(accounts) // 4)
    chosen = rng.choice(len(accounts), size=total, replace=False, p=weights)

    cursor = 0
    for r in rings:
        take = min(r["size"], total - cursor)
        if take < 3:
            r["members"] = np.zeros(0, dtype=np.int64)
            continue
        r["members"] = chosen[cursor:cursor + take].astype(np.int64)
        r["size"] = take
        cursor += take
    return [r for r in rings if len(r.get("members", [])) >= 3]


def _assign_devices(cfg, accounts, ring, rng):
    """Sloppy rings share a handful of devices. Careful rings mostly don't.

    Kept sparse on purpose: if every ring shared a device, that single
    feature would carry the whole graph result.
    """
    members = ring["members"]
    if ring["difficulty"] == "blatant":
        pool = accounts.loc[members[:cfg.blatant["device_pool"]], "device_id"].to_numpy()
        accounts.loc[members, "device_id"] = rng.choice(pool, size=len(members))
    else:
        for i in range(0, len(members) - 1, 2):
            if rng.random() < cfg.subtle_ring_device_share:
                accounts.loc[members[i + 1], "device_id"] = accounts.loc[members[i], "device_id"]


def generate_rings(cfg, rng, reg, buf, accounts, rings):
    """Emit ring transactions and stamp labels onto the accounts table."""
    for ring in rings:
        members = ring["members"]
        difficulty = ring["difficulty"]
        policy = cfg.blatant if difficulty == "blatant" else cfg.subtle

        # last 1-2 members act as consolidators; everyone else is a collector
        n_cons = 1 if len(members) < 12 else 2
        consolidators = members[-n_cons:]
        collectors = members[:-n_cons]
        exits = reg.new_many("cash_out", rng.integers(1, 3))

        accounts.loc[members, "is_mule"] = True
        accounts.loc[members, "ring_id"] = ring["ring_id"]
        accounts.loc[members, "mule_difficulty"] = difficulty
        _assign_devices(cfg, accounts, ring, rng)

        cons_inflow = {int(c): [] for c in consolidators}

        for m in collectors:
            act_day = int(min(cfg.n_days - 4,
                              ring["start_day"] + rng.integers(0, policy["stagger_days"] + 1)))
            accounts.loc[m, "activation_day"] = act_day
            active_days = cfg.n_days - act_day
            if active_days < 3:
                continue

            spd = int(rng.integers(*policy["senders_per_day"]))
            target = ring["per_account_throughput"] * rng.uniform(0.8, 1.2)
            per_credit = target / max(1, active_days * spd)

            days = np.arange(act_day, cfg.n_days)
            counts = rng.poisson(spd, size=days.size).clip(0, None)
            n_credits = int(counts.sum())
            if n_credits == 0:
                continue

            # Most victims are strangers who never send again -- the
            # structural opposite of a shopkeeper's repeat customer base.
            # But fraudsters routinely split one large transfer into 2-3
            # sends to stay under per-transaction limits, so a minority of
            # victims DO repeat. Without this the repeat-rate feature is a
            # perfect separator and the model learns the generator, not fraud.
            n_distinct = max(1, int(n_credits * rng.uniform(0.62, 0.85)))
            victim_pool = reg.new_many("victim", n_distinct)
            victims = rng.choice(victim_pool, size=n_credits, replace=True)
            days_rep = np.repeat(days, counts)
            ts_in = (days_rep * DAY + rng.integers(8 * 60, 22 * 60, n_credits)).astype(np.int32)
            amts_in = _round_ish(rng, np.maximum(
                rng.lognormal(np.log(per_credit), 0.45, n_credits), 500.0))
            buf.add(ts_in, victims, m, amts_in, CH_UPI)

            # forward the day's take in one or two batches after a delay
            keep = rng.uniform(*policy["keep_fraction"])
            order = np.argsort(ts_in, kind="stable")
            ts_s, amt_s, day_s = ts_in[order], amts_in[order], days_rep[order]
            uniq_days, idx = np.unique(day_s, return_index=True)
            splits = np.split(np.arange(ts_s.size), idx[1:])

            f_ts, f_amt = [], []
            for sl in splits:
                if sl.size == 0:
                    continue
                delay = int(rng.integers(*policy["drain_delay_min"]))
                t = int(ts_s[sl].max()) + delay
                # A 4-10 hour delay applied to a credit that arrived at 10pm
                # lands the forward at 4am. That made "fraction of activity at
                # night" a near-perfect mule detector (AUC 0.93) -- the model
                # was learning the simulator's clock, not fraud. Real operators
                # mostly move money when they are awake, so pushes that land in
                # the dead of night usually wait for morning.
                hour = (t % DAY) // 60
                if 1 <= hour < 6 and rng.random() < 0.8:
                    t = (t // DAY) * DAY + int(rng.integers(7 * 60, 11 * 60))
                    if t <= int(ts_s[sl].max()):
                        t += DAY
                if t >= cfg.horizon_min:
                    continue
                f_ts.append(t)
                f_amt.append(float(amt_s[sl].sum() * (1.0 - keep)))
            if f_ts:
                cons = int(rng.choice(consolidators))
                f_ts = np.array(f_ts, dtype=np.int32)
                f_amt = np.array(f_amt, dtype=np.float64)
                buf.add(f_ts, m, cons, f_amt, CH_IMPS)
                cons_inflow[cons].append((f_ts, f_amt))

            # Subtle mules buy groceries and pay a phone bill so the account
            # never looks purely mechanical. Crucially this spending comes
            # OUT OF THE CUT THEY KEEP -- a pipe cannot spend money it already
            # forwarded. Modelling noise as an independent stream pushed
            # pass-through above 1.0, which is physically impossible and made
            # the feature trivially separable.
            kept = float(amts_in.sum() * keep)
            spend_frac = rng.uniform(*policy["noise_spend_fraction"])
            noise_budget = kept * spend_frac
            n_noise = int(np.clip(round(active_days * rng.uniform(0.3, 1.1)), 0, 60))
            if n_noise > 0 and noise_budget > 200:
                merch = reg.new_many("merchant", 5)
                dn = rng.integers(act_day, cfg.n_days, n_noise)
                share = rng.dirichlet(np.ones(n_noise)) * noise_budget
                buf.add((dn * DAY + rng.integers(8 * 60, 22 * 60, n_noise)).astype(np.int32),
                        m, rng.choice(merch, size=n_noise), share, CH_UPI)

        # consolidators push everything out of the platform
        for c in consolidators:
            accounts.loc[c, "activation_day"] = ring["start_day"]
            chunks = cons_inflow[int(c)]
            if not chunks:
                continue
            all_ts = np.concatenate([t for t, _ in chunks])
            all_amt = np.concatenate([a for _, a in chunks])
            o = np.argsort(all_ts, kind="stable")
            all_ts, all_amt = all_ts[o], all_amt[o]
            cd = all_ts // DAY
            uniq, idx = np.unique(cd, return_index=True)
            splits = np.split(np.arange(all_ts.size), idx[1:])
            out_ts, out_amt = [], []
            for sl in splits:
                if sl.size == 0:
                    continue
                t = int(all_ts[sl].max()) + int(rng.integers(20, 240))
                if t >= cfg.horizon_min:
                    continue
                out_ts.append(t)
                out_amt.append(float(all_amt[sl].sum() * rng.uniform(0.94, 0.99)))
            if out_ts:
                buf.add(np.array(out_ts, dtype=np.int32), c,
                        rng.choice(exits, size=len(out_ts)),
                        np.array(out_amt), CH_NEFT)

    return accounts, rings
