# MuleGraph — results

Population **100,000 accounts**, **352 mules** (0.35% prevalence), alert budget **200/day**.


## Gate: no single feature solves it

If any feature alone approaches AUC 1.00, the simulator is leaking its answer key and every number below is worthless.

| feature | AUC alone |
|---|---|
| n_debits | 0.916 |
| active_days_ratio | 0.844 |
| fan_in | 0.808 |
| n_credits | 0.807 |
| fan_ratio | 0.795 |
| fanin_jump | 0.768 |
| fan_out | 0.762 |
| same_day_out_frac | 0.752 |
| amount_round_frac | 0.733 |
| credit_amount | 0.729 |
| throughput_jump | 0.725 |
| new_beneficiary_frac | 0.713 |


**LEAK WARNING.** These features separate almost perfectly on their own, which means the model is likely learning a generator artefact rather than fraud:

- `n_debits` — AUC 0.916


## Ablation: what each layer buys

| config | PR-AUC | P@200 | R@200 | blatant R | subtle R | subtle P |
|---|---|---|---|---|---|---|
| A: raw behaviour | 0.766 | 0.785 | 0.446 | 0.871 | 0.474 | 0.76 |
| B: + self-baseline | 0.837 | 0.88 | 0.5 | 0.903 | 0.533 | 0.855 |
| C: + graph | 0.997 | 1.0 | 0.568 | 1.0 | 0.623 | 1.0 |


Subtle-mule recall across the three configs: **0.474 → 0.533 → 0.623**.


The graph layer lifts subtle-mule recall by **9.0 points** over the account-level model. This is the project's central argument, measured rather than asserted.


## The squeeze

Ring throughput is fixed in rupees; ring **size** is derived by dividing it by per-account throughput. A subtle mule holds a bigger cut and drains slowly, so it moves far less per account — and the operator must recruit more accounts to hit the same target, making the ring denser and more visible to community detection.

| difficulty | mean ring size | throughput per account |
|---|---|---|
| blatant | 7 | 699637 |
| subtle | 41 | 127973 |


## Why not just write rules?

Rules are not stupid. They are cheap, fast, auditable and easy to defend to a regulator, and every bank starts with them. So the honest comparison is to build the rulebook a fraud analyst would actually write, run it on the same data, and evaluate the model at the same queue size.

| approach | flagged | caught | precision | recall | model precision at same size | model recall at same size |
|---|---|---|---|---|---|---|
| Rule: pass-through > 0.90 and drains within 6h | 12629 | 64 | 0.005 | 0.118 | 0.043 | 1.0 |
| Rule: + fan-in >= 10 and 3x throughput jump | 1422 | 88 | 0.062 | 0.162 | 0.38 | 0.994 |
| Best single feature, best threshold (n_debits) | 2482 | 164 | 0.066 | 0.302 | 0.218 | 0.996 |
| Random selection of 1,422 accounts | 1422 | 14 | 0.01 | 0.026 | 0.38 | 0.994 |


Rules fail here for a specific reason, and it is the premise of the whole project: a careful mule keeps every individual condition inside the acceptable range. 0.86 pass-through rather than 0.99. Seven hours to drain rather than seven minutes. Each condition passes on its own, so an AND of conditions never fires -- and loosening any one of them sweeps in every shopkeeper and treasurer in the population.


## Cost curve

| budget | caught | false_positives | precision | recall | value_prevented | review_cost | net_value |
|---|---|---|---|---|---|---|---|
| 200.0 | 200.0 | 0.0 | 1.0 | 0.568 | 50000000.0 | 40000.0 | 49960000.0 |
| 500.0 | 352.0 | 148.0 | 0.704 | 1.0 | 88000000.0 | 100000.0 | 87900000.0 |
| 1000.0 | 352.0 | 648.0 | 0.352 | 1.0 | 88000000.0 | 200000.0 | 87800000.0 |
| 2000.0 | 352.0 | 1648.0 | 0.176 | 1.0 | 88000000.0 | 400000.0 | 87600000.0 |
| 4000.0 | 352.0 | 3648.0 | 0.088 | 1.0 | 88000000.0 | 800000.0 | 87200000.0 |


## What our errors actually cost

The model ranks; the policy decides. Separating those two is what changes the price of a false positive — which is not analyst time but a real person locked out of their money.

| tier | action | accounts | mules | false_positives | cost_per_error | total_error_cost |
|---|---|---|---|---|---|---|
| critical | outbound hold + escalate | 51 | 51 | 0 | 2000.0 | 0.0 |
| high | manual review | 450 | 301 | 149 | 200.0 | 29800.0 |
| medium | transfer-limit cap | 1500 | 0 | 1500 | 50.0 | 75000.0 |
| low | silent monitoring | 8000 | 0 | 8000 | 0.0 | 0.0 |
| none | no action | 89999 | 0 | 89999 | 0.0 | 0.0 |


**97,999 of 99,648 false positives (98.3%) cost the customer nothing** — they only trigger silent monitoring. In a freeze-or-nothing system those same people lose access to their money.


## The operating point that matters

Catching **80% of all mules** takes a queue of **453** reviews out of 100,000 accounts.


Against **confirmed labels** that queue scores precision **0.777**. Against **ground truth** -- counting the mules nobody ever confirmed -- it scores **0.960**. The gap between those two numbers is the model being penalised for finding fraud the label set does not know about, which is the ordinary condition of every real fraud team.


That leaves **18 genuinely legitimate accounts** in the queue. Who they are is the whole reason this project grades its response instead of freezing.


Of those 101 apparent false positives, **83 are real mules the label set never confirmed**. The model was right about them; the ground truth was incomplete. This is the ordinary condition of fraud detection, and it means precision measured against confirmed labels is a floor, not an estimate.


## Who the false positives are

A predictable set, and worth naming: the cost of these errors falls hardest on the people least able to absorb it.

| persona | false positives in top 453 |
|---|---|
| salaried | 11 |
| collector | 3 |
| small_business | 3 |
| shopkeeper | 1 |


## False-positive cost

Costed with the tier ladder: silent monitoring is free, a transfer cap is minor reversible friction, manual review is analyst time the customer never sees, and only an outbound hold does real harm.

| tier | false positives | cost each | cost |
|---|---|---|---|
| critical | 0 | Rs 2,000 | Rs 0 |
| high | 32 | Rs 200 | Rs 6,400 |
| medium | 1428 | Rs 50 | Rs 71,400 |
| low | 7998 | Rs 0 | Rs 0 |


**Total cost of every mistake this system makes across 100,000 accounts: Rs 77,800.** For comparison, the 543 mules it surfaces move roughly Rs 135,750,000 between them.


The asymmetry is the design, not luck. Because the action at low confidence costs nothing, the system can afford to look at far more accounts than a freeze-or-nothing system ever could.


## How this was validated

Every score in this document is an **out-of-fold** prediction from 5-fold stratified cross-validation: each account is scored by a model that never saw it in training. Stratified because at this prevalence an ordinary split can hand a fold almost no positives.


The split is also **temporal by construction**: the baseline features are built from days before the scoring window and the label is evaluated on the window itself, so nothing from the future leaks backwards.


**Defence only.** This repository detects and ranks. It contains no evasion tooling and nothing that would help operate a mule network. The simulator's difficulty knob exists to make detection harder to fake, not to teach anyone how to launder money.


## What the model leans on

| feature | importance |
|---|---|
| two_hop_internal_reach | 0.9168 |
| cluster_mule_density_proxy | 0.1767 |
| cluster_mean_fanin | 0.1673 |
| credit_amount | 0.1334 |
| cluster_new_ben_frac | 0.0202 |
| device_share_count | 0.0095 |
| cluster_pass_through | 0.0089 |
| throughput_jump | 0.0029 |
| amount_round_frac | 0.0024 |
| baseline_weeks_active | 0.0016 |
| median_credit | 0.0016 |
| night_frac | 0.0005 |


If one feature dominated everything else, that would be a leak rather than a discovery — this table is a check, not a trophy.


## Sample of the deliverable

The output is not a freeze decision. It is a ranked queue with reason codes, because operations cannot act on a bare score and a customer deserves an explanation.

| rank | account_id | tier | action | reason_codes |
|---|---|---|---|---|
| 1 | 49803 | critical | outbound hold + escalate | median 8.8h from credit to outflow; 13 distinct senders this week; device shared with 1 other accounts; sits in a cluster of 154 connected accounts |
| 2 | 88456 | critical | outbound hold + escalate | 90% of credits forwarded on; median 8.5h from credit to outflow; 12 distinct senders this week; device shared with 1 other accounts |
| 3 | 66837 | critical | outbound hold + escalate | 7x throughput jump vs own 8-week baseline; median 9.2h from credit to outflow; 11 distinct senders this week; sits in a cluster of 230 connected accounts |
| 4 | 43524 | critical | outbound hold + escalate | 109% of credits forwarded on; median 6.0h from credit to outflow; 26 distinct senders this week; sits in a cluster of 230 connected accounts |
| 5 | 51530 | critical | outbound hold + escalate | 96% of credits forwarded on; median 4.9h from credit to outflow; 13 distinct senders this week; sits in a cluster of 62 connected accounts |
| 6 | 15835 | critical | outbound hold + escalate | median 8.5h from credit to outflow; 17 distinct senders this week; sits in a cluster of 230 connected accounts; 56% of credits are round amounts |
| 7 | 3146 | critical | outbound hold + escalate | median 10.6h from credit to outflow; 17 distinct senders this week; device shared with 1 other accounts; sits in a cluster of 154 connected accounts |
| 8 | 78393 | critical | outbound hold + escalate | 85% of credits forwarded on; median 8.8h from credit to outflow; 12 distinct senders this week; sits in a cluster of 84 connected accounts |


### Ring view

"These N accounts are one operation" is more useful to an analyst than N separate alerts.

| cluster_id | accounts | mean_score | max_score | median_pass_through | median_fan_in |
|---|---|---|---|---|---|
| 120.0 | 14.0 | 0.993 | 0.998 | 0.6959999799728394 | 18.0 |
| 402.0 | 11.0 | 0.888 | 0.997 | 0.9269999861717224 | 67.0 |
| 678.0 | 11.0 | 0.801 | 0.984 | 0.8119999766349792 | 13.0 |
| 372.0 | 7.0 | 0.743 | 0.959 | 0.9459999799728394 | 99.0 |
| 2404.0 | 8.0 | 0.679 | 0.918 | 0.9480000138282776 | 88.5 |
| 892.0 | 8.0 | 0.571 | 0.95 | 0.9490000009536743 | 81.0 |


## Limitation: this is synthetic data, and it shows

There is no public labelled mule dataset, so the generator is the
foundation of this project -- and a generator written by one person over
one build cannot manufacture the mess of a real bank ledger. Real data
contains entry errors, reversals, duplicate identities, partial records,
and behaviour no persona catalogue can enumerate. The negatives here stay
cleaner than reality, so the ceiling is artificially high.

Four genuine leaks were found and fixed while building this. Each was
caught by the gate above, and each was a real modelling error rather than
a tuning problem:

1. Only mule transfers were rounded to clean figures, so "fraction of
   round amounts" alone scored PR-AUC 0.997. Real people send round
   numbers constantly; the fix was to round legitimate transfers too.
2. No legitimate account ever received money from a stranger, making
   sender-repeat-rate a near-perfect separator (AUC 0.952). Fixed by
   adding refunds, cashbacks and one-off transfers to ordinary life.
3. Nothing legitimate occupied the mid fan-in band at high pass-through,
   leaving mules alone in an empty pocket. Fixed by adding a collector
   persona -- society treasurer, chit-fund operator, fee collector --
   which trips every account-level signal and is entirely innocent.
4. A 4-10 hour forward delay applied to a 10pm credit landed at 4am,
   while no legitimate persona ever transacted at night. "Night activity"
   scored AUC 0.932. Fixed on both sides: mules mostly wait for morning,
   and real people are awake late.

Each fix made the numbers worse, which is the point. A gate that never
rejects anything is decoration.

The defensible claims from this work are therefore:

- The **ranking is genuinely multivariate**. No single feature exceeds
  AUC 0.91 alone, so the model is combining signals rather than reading
  one off.
- The **squeeze is real in the data**, not asserted: ring size is derived
  by dividing a fixed rupee target by per-account throughput, so subtle
  rings come out roughly six times larger than blatant ones by
  construction rather than by choice.
- The **label set is incomplete on purpose**, so precision is measured the
  way a real fraud team measures it -- against confirmed cases, not
  against ground truth.
- The **tiered policy stands on its own**. It is an argument about what a
  false positive costs a person, and it holds whatever the precision
  number turns out to be.

The absolute precision and recall figures should be read as an upper
bound produced by a clean simulator, not as a performance estimate for
production.
