"""Normal-account transaction generators.

Design rule: these emit *transactions*, never features. No generator here
knows what "pass-through ratio" is.

Design rule 2: several personas are deliberately built to look like mules.
A shopkeeper sweeping the day's UPI take to a supplier has near-total
pass-through, a fast drain, and enormous fan-in. If our negatives were
clean, the model would hit 0.99 precision and prove nothing. The overlap
is the point -- what separates them is sender *repeat* behaviour,
beneficiary stability, and the fact that the shopkeeper has done this
since week 1.
"""

import numpy as np

from .population import CH_UPI, CH_IMPS, CH_NEFT, CH_CARD

DAY = 1440


def _times(rng, days, counts, lo_h, hi_h):
    """Timestamps in minutes for `counts[i]` events on `days[i]`."""
    days_rep = np.repeat(days, counts)
    if days_rep.size == 0:
        return np.zeros(0, dtype=np.int32)
    mins = rng.integers(lo_h * 60, hi_h * 60, size=days_rep.size)
    return (days_rep * DAY + mins).astype(np.int32)


def _lognorm(rng, median, sigma, n):
    return np.maximum(rng.lognormal(np.log(median), sigma, size=n), 20.0)


def _pool_draw(rng, pool, n, repeat_p):
    """Draw n counterparties from a pool, with `repeat_p` controlling how
    concentrated the draw is. High repeat_p -> the same faces keep coming
    back, which is what a real customer base looks like."""
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = max(1, int(len(pool) * (1.0 - repeat_p)))
    active = rng.choice(pool, size=min(k, len(pool)), replace=False)
    return rng.choice(active, size=n, replace=True)


# ----------------------------------------------------------------------
# personas
# ----------------------------------------------------------------------

def gen_salaried(cfg, rng, reg, buf, acct, employer_hubs):
    """One big credit a month, spending spread out, real retention."""
    employer = int(rng.choice(employer_hubs))
    landlord = reg.new("landlord")
    merchants = reg.new_many("merchant", 12)

    salary_days = np.arange(1, cfg.n_days, 30)
    salary = float(_lognorm(rng, 55_000, 0.45, 1)[0])
    ts_sal = (salary_days * DAY + rng.integers(9 * 60, 12 * 60, salary_days.size)).astype(np.int32)
    buf.add(ts_sal, employer, acct, np.full(salary_days.size, salary), CH_NEFT)

    # rent: fixed beneficiary, a couple of days after payday
    rent = salary * rng.uniform(0.22, 0.34)
    ts_rent = ts_sal + rng.integers(2 * DAY, 4 * DAY, ts_sal.size)
    ts_rent = ts_rent[ts_rent < cfg.horizon_min]
    buf.add(ts_rent, acct, landlord, np.full(ts_rent.size, rent), CH_IMPS)

    # everyday spending from a stable personal merchant set
    n_spend = rng.poisson(24 * cfg.weeks / 4.3)
    days = rng.integers(0, cfg.n_days, n_spend)
    ts = _times(rng, days, np.ones(n_spend, dtype=int), 7, 23)
    amts = _lognorm(rng, salary * 0.012, 0.8, n_spend)
    buf.add(ts, acct, _pool_draw(rng, merchants, n_spend, 0.5), amts, CH_CARD)


def gen_student(cfg, rng, reg, buf, acct, _hubs):
    """HARD NEGATIVE. Parent's money arrives and leaves within hours.

    High pass-through, fast drain, changepoint-free. Only fan-in saves it:
    one or two senders, every month, forever.
    """
    parents = reg.new_many("family", rng.integers(1, 3))
    landlord = reg.new("landlord")
    college = reg.new("institution")
    merchants = reg.new_many("merchant", 8)

    credit_days = np.arange(2, cfg.n_days, 30)
    amt = float(_lognorm(rng, 22_000, 0.4, 1)[0])
    ts_in = (credit_days * DAY + rng.integers(8 * 60, 20 * 60, credit_days.size)).astype(np.int32)
    buf.add(ts_in, rng.choice(parents), acct, np.full(credit_days.size, amt), CH_UPI)

    # rent goes out the same day, 2-10 hours later
    ts_rent = ts_in + rng.integers(120, 600, ts_in.size)
    ts_rent = ts_rent[ts_rent < cfg.horizon_min]
    buf.add(ts_rent, acct, landlord, np.full(ts_rent.size, amt * rng.uniform(0.45, 0.62)), CH_UPI)

    if rng.random() < 0.5:
        buf.add(np.array([ts_in[0] + DAY * 3]), acct, college,
                np.array([amt * rng.uniform(0.6, 1.4)]), CH_NEFT)

    n_spend = rng.poisson(9 * cfg.weeks / 4.3)
    days = rng.integers(0, cfg.n_days, n_spend)
    ts = _times(rng, days, np.ones(n_spend, dtype=int), 8, 23)
    buf.add(ts, acct, _pool_draw(rng, merchants, n_spend, 0.5),
            _lognorm(rng, 320, 0.7, n_spend), CH_UPI)


def gen_shopkeeper(cfg, rng, reg, buf, acct, _hubs):
    """HARDEST NEGATIVE. Trips every account-level signal we have.

    Dozens of small credits a day from people the bank has never linked to
    this account, swept almost entirely to a supplier each night. That is
    high fan-in, low fan-out, ~0.9 pass-through and a same-day drain --
    a mule on paper. The tells are that the customers repeat, the supplier
    never changes, and it has looked exactly like this since day one.
    """
    customers = reg.new_many("retail_customer", 260)
    suppliers = reg.new_many("supplier", rng.integers(1, 3))
    merchants = reg.new_many("merchant", 6)

    days = np.arange(cfg.n_days)
    counts = rng.poisson(rng.uniform(5, 34), size=cfg.n_days)
    counts[days % 7 == 6] = (counts[days % 7 == 6] * 0.4).astype(int)  # Sunday dip

    ts_in = _times(rng, days, counts, 9, 21)
    amts_in = _lognorm(rng, 380, 0.85, ts_in.size)
    buf.add(ts_in, _pool_draw(rng, customers, ts_in.size, 0.62), acct, amts_in, CH_UPI)

    # nightly sweep of most of the day's take to a fixed supplier
    take = np.zeros(cfg.n_days)
    np.add.at(take, ts_in // DAY, amts_in)
    sweep_frac = rng.uniform(0.82, 0.94)
    active = take > 0
    ts_sweep = (days[active] * DAY + rng.integers(21 * 60, 23 * 60, active.sum())).astype(np.int32)
    buf.add(ts_sweep, acct, rng.choice(suppliers, size=active.sum()),
            take[active] * sweep_frac, CH_IMPS)

    n_spend = rng.poisson(12 * cfg.weeks / 4.3)
    days_s = rng.integers(0, cfg.n_days, n_spend)
    buf.add(_times(rng, days_s, np.ones(n_spend, dtype=int), 8, 22), acct,
            _pool_draw(rng, merchants, n_spend, 0.5),
            _lognorm(rng, 600, 0.7, n_spend), CH_CARD)


def gen_freelancer(cfg, rng, reg, buf, acct, _hubs):
    """Irregular, multi-payer income with genuine client churn."""
    clients = reg.new_many("client", 9)
    merchants = reg.new_many("merchant", 10)
    landlord = reg.new("landlord")

    n_inv = rng.poisson(1.4 * cfg.weeks)
    days = np.sort(rng.integers(0, cfg.n_days, n_inv))
    ts_in = _times(rng, days, np.ones(n_inv, dtype=int), 10, 19)
    amts = _lognorm(rng, 28_000, 0.7, n_inv)
    buf.add(ts_in, _pool_draw(rng, clients, n_inv, 0.45), acct, amts, CH_NEFT)

    rent_days = np.arange(4, cfg.n_days, 30)
    buf.add((rent_days * DAY + 11 * 60).astype(np.int32), acct, landlord,
            np.full(rent_days.size, float(_lognorm(rng, 14_000, 0.35, 1)[0])), CH_UPI)

    n_spend = rng.poisson(16 * cfg.weeks / 4.3)
    days_s = rng.integers(0, cfg.n_days, n_spend)
    buf.add(_times(rng, days_s, np.ones(n_spend, dtype=int), 8, 23), acct,
            _pool_draw(rng, merchants, n_spend, 0.4),
            _lognorm(rng, 1_200, 0.9, n_spend), CH_CARD)


def gen_small_business(cfg, rng, reg, buf, acct, _hubs, internal_pool=None):
    """Customer fan-in plus vendor and payroll fan-out.

    The payroll leg pays *internal* accounts, which gives the graph layer
    legitimate dense communities to trip over. Without these, community
    detection would find only mule rings and the graph result would be a
    circular artefact of the generator.
    """
    customers = reg.new_many("retail_customer", 130)
    vendors = reg.new_many("supplier", rng.integers(3, 7))

    days = np.arange(cfg.n_days)
    counts = rng.poisson(rng.uniform(5, 12), size=cfg.n_days)
    ts_in = _times(rng, days, counts, 9, 20)
    amts_in = _lognorm(rng, 2_400, 0.9, ts_in.size)
    buf.add(ts_in, _pool_draw(rng, customers, ts_in.size, 0.55), acct, amts_in, CH_UPI)

    weekly = np.arange(3, cfg.n_days, 7)
    for v in vendors:
        buf.add((weekly * DAY + rng.integers(10 * 60, 18 * 60, weekly.size)).astype(np.int32),
                acct, int(v), _lognorm(rng, 22_000, 0.5, weekly.size), CH_NEFT)

    if internal_pool is not None and len(internal_pool) > 0:
        staff = rng.choice(internal_pool, size=rng.integers(2, 7), replace=False)
        pay_days = np.arange(1, cfg.n_days, 30)
        for s in staff:
            buf.add((pay_days * DAY + 12 * 60).astype(np.int32), acct, int(s),
                    _lognorm(rng, 18_000, 0.3, pay_days.size), CH_NEFT)


def gen_low_activity(cfg, rng, reg, buf, acct, _hubs):
    """Mostly dormant. This is also the pool mules get recruited from."""
    others = reg.new_many("family", 3)
    merchants = reg.new_many("merchant", 5)

    n_in = rng.poisson(2.0 * cfg.weeks / 4.3)
    days = rng.integers(0, cfg.n_days, n_in)
    buf.add(_times(rng, days, np.ones(n_in, dtype=int), 9, 21),
            _pool_draw(rng, others, n_in, 0.6), acct,
            _lognorm(rng, 4_000, 0.8, n_in), CH_UPI)

    n_out = rng.poisson(3.0 * cfg.weeks / 4.3)
    days_o = rng.integers(0, cfg.n_days, n_out)
    buf.add(_times(rng, days_o, np.ones(n_out, dtype=int), 9, 22), acct,
            _pool_draw(rng, merchants, n_out, 0.5),
            _lognorm(rng, 900, 0.8, n_out), CH_UPI)


def gen_micro_merchant(cfg, rng, reg, buf, acct, _hubs):
    """Home business: tiffin, tailoring, reselling.

    Exists to populate the fan-in band that subtle mules occupy. Without
    this persona there is an empty gap between "student with one sender"
    and "shopkeeper with seventy", and the model gets a free separator
    that would never exist in real data.
    """
    customers = reg.new_many("retail_customer", 70)
    supplier = reg.new("supplier")
    merchants = reg.new_many("merchant", 8)

    days = np.arange(cfg.n_days)
    counts = rng.poisson(rng.uniform(3, 18), size=cfg.n_days)
    ts_in = _times(rng, days, counts, 9, 21)
    amts_in = _lognorm(rng, 700, 0.8, ts_in.size)
    buf.add(ts_in, _pool_draw(rng, customers, ts_in.size, 0.50), acct, amts_in, CH_UPI)

    # restocks a couple of times a week, keeps a real margin
    take = np.zeros(cfg.n_days)
    np.add.at(take, ts_in // DAY, amts_in)
    restock_days = np.arange(rng.integers(0, 3), cfg.n_days, rng.integers(2, 5))
    frac = rng.uniform(0.45, 0.72)
    for d in restock_days:
        lo = max(0, d - 4)
        amt = take[lo:d + 1].sum() * frac
        if amt > 100:
            buf.add(np.array([d * DAY + rng.integers(10 * 60, 20 * 60)]),
                    acct, supplier, np.array([amt]), CH_IMPS)

    n_spend = rng.poisson(14 * cfg.weeks / 4.3)
    days_s = rng.integers(0, cfg.n_days, n_spend)
    buf.add(_times(rng, days_s, np.ones(n_spend, dtype=int), 8, 22), acct,
            _pool_draw(rng, merchants, n_spend, 0.5),
            _lognorm(rng, 500, 0.8, n_spend), CH_UPI)


def sprinkle_stranger_credits(cfg, rng, reg, buf, acct):
    """Everyone occasionally receives money from someone they have no
    history with: a refund, a deposit returned, a friend settling a bill,
    a marketplace payout, a cashback.

    This matters more than it looks. Without it, every legitimate account
    has a sender-repeat-rate of exactly 1.00 and that single feature
    separates mules almost perfectly -- not because it detects fraud, but
    because the simulator forgot that strangers pay real people too.
    """
    n = rng.poisson(1.6 * cfg.weeks / 4.3)
    if n == 0:
        return
    senders = reg.new_many("one_off", n)
    days = rng.integers(0, cfg.n_days, n)
    buf.add(_times(rng, days, np.ones(n, dtype=int), 8, 22),
            senders, acct, _lognorm(rng, 2_500, 1.1, n), CH_UPI)


GENERATORS = {
    "salaried": gen_salaried,
    "student": gen_student,
    "shopkeeper": gen_shopkeeper,
    "micro_merchant": gen_micro_merchant,
    "freelancer": gen_freelancer,
    "small_business": gen_small_business,
    "low_activity": gen_low_activity,
}


# ----------------------------------------------------------------------
# life events -- the false positives we care most about
# ----------------------------------------------------------------------

def apply_life_event(cfg, rng, reg, buf, acct, kind, day):
    """Overlay an event that breaks this account's own baseline."""
    if kind == "wedding":
        # Dozens of one-time senders, money out within a day or two to
        # vendors. Structurally almost identical to a subtle mule.
        n = rng.integers(30, 80)
        senders = reg.new_many("wedding_guest", n)
        ts_in = (day * DAY + rng.integers(0, 3 * DAY, n)).astype(np.int32)
        amts = _lognorm(rng, 8_000, 0.9, n)
        buf.add(ts_in, senders, acct, amts, CH_UPI)
        vendors = reg.new_many("event_vendor", rng.integers(5, 11))
        total = amts.sum() * rng.uniform(0.85, 0.97)
        share = rng.dirichlet(np.ones(len(vendors))) * total
        ts_out = (day * DAY + rng.integers(12 * 60, 4 * DAY, len(vendors))).astype(np.int32)
        buf.add(ts_out, acct, vendors, share, CH_IMPS)

    elif kind == "medical":
        n = rng.integers(4, 14)
        senders = reg.new_many("family", n)
        ts_in = (day * DAY + rng.integers(0, 2 * DAY, n)).astype(np.int32)
        amts = _lognorm(rng, 25_000, 0.7, n)
        buf.add(ts_in, senders, acct, amts, CH_IMPS)
        hospital = reg.new("institution")
        buf.add(np.array([day * DAY + DAY]), acct, hospital,
                np.array([amts.sum() * rng.uniform(0.88, 0.99)]), CH_NEFT)

    elif kind == "job_change":
        new_emp = reg.new("employer")
        pay_days = np.arange(day, cfg.n_days, 30)
        if pay_days.size:
            buf.add((pay_days * DAY + 10 * 60).astype(np.int32), new_emp, acct,
                    _lognorm(rng, 68_000, 0.4, pay_days.size), CH_NEFT)

    elif kind == "city_move":
        # every beneficiary is suddenly new -- and income follows the person,
        # otherwise this account would show impossible outflow with no inflow
        new_emp = reg.new("employer")
        pay = np.arange(day, cfg.n_days, 30)
        if pay.size:
            buf.add((pay * DAY + 10 * 60).astype(np.int32), new_emp, acct,
                    _lognorm(rng, 60_000, 0.4, pay.size), CH_NEFT)
        new_ben = reg.new_many("merchant", rng.integers(6, 14))
        n = rng.integers(15, 40)
        days_o = rng.integers(day, cfg.n_days, n)
        buf.add(_times(rng, days_o, np.ones(n, dtype=int), 8, 22), acct,
                rng.choice(new_ben, size=n), _lognorm(rng, 3_500, 0.9, n), CH_UPI)
