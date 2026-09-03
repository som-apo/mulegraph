"""Central configuration for MuleGraph.

Everything the simulator can vary lives here. Nothing downstream of the
simulator is allowed to read these values -- feature code must measure
behaviour from the transaction ledger, never look up how it was generated.
"""

from dataclasses import dataclass, field

MINUTES_PER_DAY = 1440


@dataclass
class SimConfig:
    # ---- scale -------------------------------------------------------
    n_accounts: int = 20_000
    weeks: int = 9                    # weeks 1..8 baseline, week 9 = scoring window
    baseline_weeks: int = 8
    seed: int = 7

    # ---- normal population -------------------------------------------
    # These personas are chosen so that several of them are *hard negatives*:
    # a shopkeeper trips pass-through, drain speed, and fan-in all at once.
    persona_mix: dict = field(default_factory=lambda: {
        "salaried":       0.30,
        "student":        0.11,
        "shopkeeper":     0.07,
        "micro_merchant": 0.07,   # tiffin/tailoring/resale: fills the fan-in gap
        "collector":      0.05,   # treasurer/chit-fund: the hardest negative
        "freelancer":     0.11,
        "small_business": 0.07,
        "low_activity":   0.22,
    })

    # Life events break an account's own baseline -- the same changepoint
    # signature a freshly-activated mule produces.
    life_event_rate: float = 0.08
    life_event_mix: dict = field(default_factory=lambda: {
        "wedding":     0.30,   # dozens of one-time senders, fast outflow. Worst FP.
        "medical":     0.20,
        "job_change":  0.30,
        "city_move":   0.20,
    })

    # ---- mule rings ---------------------------------------------------
    mule_rate: float = 0.005          # 0.5% prevalence, as in the design doc
    subtle_ring_fraction: float = 0.60

    # THE SQUEEZE.
    # Ring throughput target is fixed in rupees. Ring SIZE is derived from it.
    # A subtle mule moves less money per account (holds a cut, drains slowly),
    # so hitting the same target forces the operator to recruit more accounts,
    # which makes the ring denser and more visible to the graph layer.
    # This makes the project's central thesis a property of the data rather
    # than an assertion on a slide.
    ring_throughput_min: float = 2_500_000
    ring_throughput_max: float = 6_000_000
    blatant_account_throughput: tuple = (500_000, 900_000)
    subtle_account_throughput: tuple = (80_000, 150_000)

    # Per-difficulty behavioural policy. Note we specify *behaviour*
    # (how fast, how many senders, how much they keep) -- never a feature value.
    blatant: dict = field(default_factory=lambda: {
        "drain_delay_min": (5, 60),        # minutes from credit to forward
        "senders_per_day": (8, 25),
        "keep_fraction": (0.01, 0.04),
        "noise_spend_fraction": (0.02, 0.10),  # share of the kept cut
        "device_pool": 2,                  # whole ring on 1-2 devices
        "stagger_days": 1,                 # everyone activates together
    })
    subtle: dict = field(default_factory=lambda: {
        "drain_delay_min": (240, 600),     # 4-10 hours
        "senders_per_day": (2, 5),
        "keep_fraction": (0.10, 0.18),
        "noise_spend_fraction": (0.35, 0.75),  # spends most of the cut - looks alive
        "device_pool": 0,                  # pairs only, assigned separately
        "stagger_days": 12,                # staggered activation
    })

    # Which personas get recruited. Rented pipes come from people who need
    # money, not from salaried customers with stable inflows.
    recruit_weights: dict = field(default_factory=lambda: {
        "student":        0.32,
        "low_activity":   0.32,
        "freelancer":     0.17,
        "micro_merchant": 0.08,
        "collector":      0.02,
        "salaried":       0.03,
        "shopkeeper":     0.03,
        "small_business": 0.03,
    })

    # ---- shared infrastructure ----------------------------------------
    # Deliberately sparse. If every ring shared a device, that one feature
    # would do all the work and the graph layer would be fake.
    normal_device_share_rate: float = 0.03   # families sharing a phone
    subtle_ring_device_share: float = 0.35   # only some subtle accounts pair up

    # In production, labels come from confirmed cases: a complaint was
    # filed, an investigation ran, the account was proven to be a mule.
    # Plenty of mules are never confirmed, so a fraction of the true
    # positives sit in the data labelled as ordinary customers. Ignoring
    # this is the single biggest way a synthetic fraud dataset ends up
    # easier than the real thing -- the model gets a clean answer key that
    # no fraud team has ever had.
    unlabeled_ring_fraction: float = 0.40

    out_dir: str = "data"

    @property
    def n_days(self) -> int:
        return self.weeks * 7

    @property
    def horizon_min(self) -> int:
        return self.n_days * MINUTES_PER_DAY

    @property
    def scoring_window_start_min(self) -> int:
        return self.baseline_weeks * 7 * MINUTES_PER_DAY
