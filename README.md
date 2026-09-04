# MuleGraph

Money mule detection on a simulated payments network. Some accounts are
rented pipes -- valid KYC, fraud money passing through. MuleGraph catches
them by behaviour and by ring structure, then hands out a ranked daily
review queue with reason codes.

**[Open the live review queue](https://som-apo.github.io/mulegraph/dashboard.html)**
· [Full results](RESULTS.md)
· [What broke at 2am, and how we got out](WHAT_BROKE.md)

Mule accounts are where merchant fraud money lands - fake seller payouts,
stolen-card refunds, gig cash-outs. Catching the ring is catching the loss,
and catching it before a complaint exists rather than after the account
already holds nothing.

## Headline numbers

100,000 accounts, 543 mules (0.54%), 9 weeks of transactions.

| | |
|---|---|
| Catching 80% of all mules | queue of 453 reviews |
| Precision vs **confirmed labels** | 0.777 |
| Precision vs **ground truth** | 0.960 |
| Genuinely legitimate accounts in that queue | 18 |
| Total cost of every mistake, across 100k accounts | Rs 77,800 |
| False positives that cost the customer nothing | 7,998 of 9,458 |
| False positives at the outbound-hold tier | 0 |

The gap between 0.777 and 0.960 is the model being penalised for finding
fraud the label set does not know about. 40% of mule rings are deliberately
left unlabelled, because no fraud team has ever had a complete answer key.

## The problem

KYC is valid, so identity checks miss these accounts entirely. The tell is
behaviour -- and a careful mule keeps every individual signal inside the
acceptable range. 0.86 pass-through instead of 0.99. Seven hours to drain
instead of seven minutes. Each feature passes on its own.

Measured on this data, no single feature exceeds **AUC 0.91** alone. That is
why this is a learned model rather than a rulebook.

Nothing separates a mule from a shopkeeper sweeping the day's UPI take to a
supplier, or a society treasurer collecting monthly dues, on any one signal.
Both are in the dataset on purpose.

## The squeeze

Ring throughput is fixed in rupees; ring **size is derived from it**:

    ring_size = ceil(ring_throughput / per_account_throughput)

A subtle mule keeps a larger cut and drains slowly, so it moves far less per
account -- and the operator must recruit more accounts to hit the same
target, which makes the ring denser and more visible to community detection.

| difficulty | mean ring size | throughput per account |
|---|---|---|
| blatant | 7 | Rs 6.9L |
| subtle | 41 | Rs 1.3L |

Evading the account-level model pushes you into the graph-level model. Here
that is arithmetic in the generator, not a claim on a slide.

## What a false positive actually costs

Not analyst time. A person locked out of their own money -- and a
predictable set of people: students, shopkeepers, treasurers, freelancers,
anyone who just moved city or had a wedding. Not salaried customers with
stable inflows.

So the model ranks and a separate policy decides:

| risk band | action | cost if wrong |
|---|---|---|
| low | silent monitoring | **Rs 0** -- customer never knows |
| medium | transfer-limit cap | minor, reversible |
| high | manual review | ~Rs 200 analyst time, invisible to the customer |
| critical | outbound hold + escalate | real harm -- high confidence only |

Because the action at low confidence is free, the system can afford to look
at far more accounts than a freeze-or-nothing system ever could. Every alert
carries reason codes: *"11x throughput jump, 87% new beneficiaries, device
shared with 5 accounts."*

## Honesty

- **The gate rejected this dataset four times.** Round amounts, sender
  repeat rate, an empty fan-in band, and night-hour activity each let a
  single feature separate mules almost perfectly. All four were real
  modelling bugs, all four are documented in RESULTS.md, and each fix made
  the numbers worse.
- **Metrics were written before the model existed**, so there was nothing to
  metric-shop for. Accuracy and ROC-AUC are refused and the reasons are in
  the code.
- **This is synthetic data and it shows.** The numbers are an upper bound
  from a clean simulator, not a production estimate.

**Defence only.** This repository detects and ranks. It contains no evasion
tooling and nothing that would help operate a mule network.

## Run it

    pip install -r requirements.txt
    python scripts/run_pipeline.py --accounts 100000
    python scripts/make_results.py       # regenerates RESULTS.md
    python -m mulegraph.policy.tiers     # ranked queue + ring view
    python scripts/inspect_overlap.py    # the leak gate

Validation is 5-fold stratified cross-validation: every account is scored by
a model that never saw it. The split is temporal by construction -- baseline
features come from days before the scoring window, the label is evaluated on
the window itself.

## Layout

    mulegraph/
      config.py            every simulation knob, including the squeeze
      sim/                 population, personas, rings, orchestrator
      features/account.py  24 behavioural features + 8-week self-baseline
      graph/features.py    money-flow graph, communities, cluster features
      models/account.py    gradient boosting + the A/B/C ablation
      eval/metrics.py      PR-AUC, P@k, R@budget, cost curve
      policy/tiers.py      risk bands, reason codes, ranked queue
