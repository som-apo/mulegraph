# What broke, and how we got out

The organisers said they read this instead of a resume. So here it is,
honestly, in the order it happened.

The short version: **we built a gate whose only job was to reject our own
dataset, and it rejected it five times.** Every rejection was a real
modelling bug. Every fix made the numbers worse. That is the whole story.

---

## The setup

There is no public labelled money-mule dataset. You cannot download one.
So the simulator is not a shortcut around the hard part -- it *is* the hard
part, and it comes with a trap: if you generate mules using a rule and then
extract that rule as a feature, the model just recovers your generator and
reports a meaningless number.

So before writing any model, we wrote `scripts/inspect_overlap.py`. It
computes each feature on its own and asks one question: does any single one
separate mules almost perfectly? If yes, the simulator is leaking its
answer key and everything downstream is worthless.

Then we tried to break it.

---

## Break 1 - pass-through above 1.0

**Symptom.** PR-AUC 0.997. Permutation importance showed one feature at
0.60 and everything else at roughly zero: `amount_round_frac`.

**Cause.** We rounded 55% of victim transfers to clean multiples of 500 so
they would look like real transfers. Our legitimate personas drew from a
lognormal, which essentially never lands on an exact multiple. So "is this
amount round?" was a near-perfect mule detector.

Digging further, subtle mules showed a pass-through ratio of **1.28** --
sending out more money than came in, which is physically impossible for a
pipe. The noise spending we gave them to look alive was modelled as an
independent stream instead of coming out of the cut they keep.

**Fix.** Real people send round numbers constantly. Rent is 12,000, a gift
is 5,100, a parent sends exactly 20,000. We rounded legitimate transfers
too, at rates that vary by transaction type -- retail spend stays messy,
P2P transfers are usually round. And mule noise spending now comes out of
the kept cut.

**Result.** `amount_round_frac` fell from a near-perfect separator to AUC
0.693. Pass-through landed at 0.91, sitting exactly on top of the
shopkeepers.

---

## Break 2 - nobody ever received money from a stranger

**Symptom.** With rounding fixed, `sender_repeat_rate` jumped to AUC 0.952.

**Cause.** Our mules received only from fresh victims who never sent again.
Meanwhile every legitimate account had a repeat rate of *exactly 1.00*,
because in our simulated world no honest person ever received money from
someone they had no history with.

**Fix.** Two sides. Fraudsters routinely split a large transfer into two or
three sends to stay under per-transaction limits, so a minority of victims
now repeat. And everybody -- salaried, student, shopkeeper -- now receives
occasional one-off credits: refunds, cashbacks, a friend settling a bill, a
marketplace payout.

**Result.** AUC 0.952 down to 0.544.

---

## Break 3 - an empty pocket in feature space

**Symptom.** Precision 1.000 at 80% recall. Zero false positives. That does
not happen.

**Cause.** This one was not a single feature, it was a hole. Subtle mules
sat at fan-in 10-30 with pass-through around 0.9. Shopkeepers had far
higher fan-in. Micro-merchants kept a real margin so their pass-through was
lower. **Nothing legitimate occupied that pocket**, so mules sat there
alone and the model got a free separator that would never exist in real
data.

**Fix.** We added the persona that actually lives there. A society
treasurer. A chit-fund operator. A tuition-fee collector. A travel agent
booking for a group. They collect from 15-40 people who have no obvious
link to each other, hold it for hours, and forward 90-97% to one
destination. Fan-in high, fan-out one, pass-through 0.93, same-day drain --
a mule on every account-level signal, and completely innocent.

What separates them is that the same members pay every cycle, and the
destination never changes.

---

## Break 4 - the model learned our clock

**Symptom.** `night_frac` at AUC 0.932.

**Cause.** Subtle mules forward money 4-10 hours after receiving it, and
credits arrive until 10pm. 22:00 plus 8 hours is 6am. Meanwhile every
legitimate persona was capped at 7am-11pm. So mules transacted at night and
nobody else ever did.

The model had not learned fraud. It had learned what time our simulator
allowed people to be awake.

**Fix.** Both sides again. Real operators mostly move money when they are
awake, so forwards landing in the dead of night usually wait until morning.
And real people order food and pay bills at 1am, so legitimate spending
windows now spill past midnight.

---

## Break 5 - the answer key was complete

**Symptom.** Even after four fixes, precision stayed at 1.000 well past 80%
recall.

**Cause.** This one was not a feature leak at all. **Every mule in our data
was labelled.** No fraud team has ever had that. Labels come from confirmed
cases: a complaint was filed, an investigation ran, the account was proven.
Plenty of mules are never confirmed and sit in the data as ordinary
customers.

**Fix.** 40% of mule rings are now hidden from the label set. They really
are mules and really do behave like mules; they are simply ones nobody
confirmed. Every one the model surfaces is scored as a false positive even
though the model was right.

**Result.** The single most interesting number in the project. At the
operating point, precision measured against **confirmed labels is 0.777**,
and against **ground truth it is 0.960**. Of 101 apparent false positives,
**83 are real mules the label set never confirmed**.

That gap is not a flaw in the model. It is the ordinary working condition
of every fraud team, and it means precision measured against confirmed
cases is a floor rather than an estimate.

---

## Two smaller ones, at about 2am

**Python 3.9.** `SimConfig | None` is 3.10+ syntax. One
`from __future__ import annotations` and it moved on.

**xgboost would not import.** `libomp.dylib` missing on Apple Silicon, and
no Homebrew on the machine. Rather than burn twenty minutes installing a
package manager at 2am, we switched to sklearn's
`HistGradientBoostingClassifier` -- same algorithm family, no OpenMP
dependency, already installed. No measurable difference to the result.

---

## What we would not do differently

Every one of those fixes made the numbers worse, and that was the correct
direction. A gate that never rejects anything is decoration.

We could have shipped after break 1 with PR-AUC 0.997 and a very confident
slide. It would have been a lie, and the first judge to ask "what is that
feature actually measuring?" would have found it in ninety seconds.

## What is still wrong

Precision remains higher than any real deployment would see. That is not a
sixth bug of the same kind -- it is the ceiling of synthetic data. Real
ledgers contain entry errors, reversals, duplicate identities, partial
records and behaviour no persona catalogue can enumerate. A generator
written over one build cannot manufacture that.

So the numbers in [RESULTS.md](RESULTS.md) are an upper bound from a clean
simulator, not a production estimate. The defensible claims are that the
ranking is genuinely multivariate, that the squeeze is arithmetic in the
data rather than an assertion, and that the tiered policy holds regardless
of where precision lands.
