"""Build a self-contained HTML console from the scored output.

    python scripts/make_dashboard.py

Writes dashboard.html. No server, no dependencies at view time -- the data
is inlined, so it opens by double-clicking and works on a projector with no
network.

This is what an analyst would actually open in the morning: the ranked
queue, why each account surfaced, and what the day's mistakes cost.
"""

from __future__ import annotations

import glob
import html
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from mulegraph.policy.tiers import assign_tiers, reason_codes, ring_view

TIER_COST = {"critical": 2000.0, "high": 200.0, "medium": 50.0,
             "low": 0.0, "none": 0.0}
TIER_ACTION = {"critical": "Outbound hold, escalate",
               "high": "Manual review",
               "medium": "Transfer-limit cap",
               "low": "Silent monitoring",
               "none": "No action"}


def load_scored():
    for ext in (".parquet", ".csv.gz"):
        hits = glob.glob(f"data/scored{ext}")
        if hits:
            return (pd.read_parquet(hits[0]) if ext == ".parquet"
                    else pd.read_csv(hits[0]))
    raise SystemExit("run scripts/run_pipeline.py first")


def rupees(x):
    return f"Rs {x:,.0f}"


def main():
    f = load_scored()
    f["tier"] = assign_tiers(f["score"].to_numpy())
    truth = (f["true_mule"].astype(bool) if "true_mule" in f.columns
             else f["is_mule"].astype(bool))
    f["confirmed"] = f["is_mule"].astype(bool)
    f["truth"] = truth

    n = len(f)
    order = np.argsort(-f["score"].to_numpy())
    yt = truth.to_numpy().astype(int)
    yc = f["confirmed"].to_numpy().astype(int)
    hits_true = np.cumsum(yt[order])
    hits_conf = np.cumsum(yc[order])
    k80 = int(np.searchsorted(hits_true, 0.80 * yt.sum()) + 1)
    p_conf = hits_conf[k80 - 1] / k80
    p_true = hits_true[k80 - 1] / k80
    innocent = k80 - int(hits_true[k80 - 1])
    unconfirmed = int(hits_true[k80 - 1] - hits_conf[k80 - 1])

    cost_rows, total_cost, free_fp, all_fp = [], 0.0, 0, 0
    for t in ["critical", "high", "medium", "low"]:
        sl = f[f["tier"] == t]
        fp = int((~sl["truth"]).sum())
        c = fp * TIER_COST[t]
        total_cost += c
        all_fp += fp
        if TIER_COST[t] == 0:
            free_fp += fp
        cost_rows.append((t, TIER_ACTION[t], len(sl), int(sl["truth"].sum()),
                          fp, TIER_COST[t], c))

    q = f.nlargest(120, "score").copy()
    q["reasons"] = q.apply(reason_codes, axis=1)
    q["rank"] = np.arange(1, len(q) + 1)

    rows = []
    for _, r in q.iterrows():
        if r["truth"] and r["confirmed"]:
            verdict, vclass = "Confirmed mule", "hit"
        elif r["truth"]:
            verdict, vclass = "Mule, never confirmed", "unconf"
        else:
            verdict, vclass = f"Legitimate - {r['persona']}", "miss"
        rows.append({
            "rank": int(r["rank"]), "acct": int(r["account_id"]),
            "score": round(float(r["score"]), 4), "tier": r["tier"],
            "action": TIER_ACTION.get(r["tier"], ""),
            "cluster": int(r["cluster_id"]) if "cluster_id" in r else -1,
            "verdict": verdict, "vclass": vclass, "reasons": r["reasons"],
        })

    rv = ring_view(f, min_size=4, top_n=8)
    ring_rows = []
    if len(rv):
        for _, r in rv.iterrows():
            sub = f[f["cluster_id"] == r["cluster_id"]]
            ring_rows.append((int(r["cluster_id"]), int(r["accounts"]),
                              int(sub["truth"].sum()),
                              round(float(r["median_pass_through"]), 2),
                              int(r["median_fan_in"])))

    fp_personas = (f.nlargest(k80, "score")
                   .pipe(lambda d: d[~d["truth"]])["persona"]
                   .value_counts().to_dict())

    css = """
:root{
  --ground:#E9EDEF; --surface:#FFFFFF; --ink:#16212B; --muted:#5D6D79;
  --rule:#C7D1D8; --rule-soft:#DEE5E9;
  --t-critical:#8C1D0E; --t-high:#C1440E; --t-medium:#B07503; --t-low:#64798A;
  --hit:#1F5F3F; --unconf:#7A5C12; --miss:#8C1D0E;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:15px;line-height:1.5;font-variant-numeric:tabular-nums;}
.wrap{max-width:1180px;margin:0 auto;padding:40px 28px 80px}
h1{font-size:30px;line-height:1.15;margin:0 0 6px;letter-spacing:-.02em;font-weight:640}
.sub{color:var(--muted);margin:0 0 34px;max-width:62ch}
h2{font-size:19px;margin:44px 0 4px;letter-spacing:-.01em;font-weight:620}
.note{color:var(--muted);margin:0 0 16px;max-width:70ch;font-size:14px}
.panel{background:var(--surface);border:1px solid var(--rule);border-radius:3px}
.split{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;
  gap:0;padding:26px 30px}
.big{font-size:52px;line-height:1;letter-spacing:-.03em;font-weight:660}
.lbl{color:var(--muted);font-size:14px;margin-top:8px;max-width:26ch}
.gap{padding:0 26px;color:var(--muted);font-size:13px;text-align:center;
  border-left:1px solid var(--rule-soft);border-right:1px solid var(--rule-soft);
  align-self:stretch;display:flex;align-items:center;max-width:20ch}
.strip{display:flex;gap:0;border-top:1px solid var(--rule);flex-wrap:wrap}
.stat{flex:1 1 0;min-width:150px;padding:16px 20px;border-right:1px solid var(--rule-soft)}
.stat:last-child{border-right:none}
.stat b{display:block;font-size:23px;font-weight:640;letter-spacing:-.02em}
.stat span{color:var(--muted);font-size:13px}
table{width:100%;border-collapse:collapse;background:var(--surface);
  border:1px solid var(--rule);border-radius:3px;overflow:hidden}
th{text-align:left;font-weight:600;font-size:13px;color:var(--muted);
  padding:10px 12px;border-bottom:1px solid var(--rule);background:#F4F7F8}
td{padding:9px 12px;border-bottom:1px solid var(--rule-soft);font-size:14px;
  vertical-align:top}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right}
.q td:first-child{border-left:3px solid transparent}
.q tr[data-tier="critical"] td:first-child{border-left-color:var(--t-critical)}
.q tr[data-tier="high"] td:first-child{border-left-color:var(--t-high)}
.q tr[data-tier="medium"] td:first-child{border-left-color:var(--t-medium)}
.q tr[data-tier="low"] td:first-child{border-left-color:var(--t-low)}
.tier{font-size:13px;font-weight:600}
.tier.critical{color:var(--t-critical)} .tier.high{color:var(--t-high)}
.tier.medium{color:var(--t-medium)} .tier.low{color:var(--t-low)}
.v{font-size:13px} .v.hit{color:var(--hit)} .v.unconf{color:var(--unconf)}
.v.miss{color:var(--miss);font-weight:600}
.reasons{color:var(--muted);font-size:13px;max-width:44ch}
.filters{display:flex;gap:8px;margin:14px 0 12px;flex-wrap:wrap}
button{font:inherit;font-size:13px;padding:6px 13px;border:1px solid var(--rule);
  background:var(--surface);color:var(--ink);border-radius:3px;cursor:pointer}
button[aria-pressed="true"]{background:var(--ink);color:var(--surface);
  border-color:var(--ink)}
button:focus-visible{outline:2px solid var(--t-high);outline-offset:2px}
.foot{margin-top:44px;padding-top:18px;border-top:1px solid var(--rule);
  color:var(--muted);font-size:13px;max-width:74ch}
@media (max-width:760px){
  .split{grid-template-columns:1fr;gap:20px}
  .gap{border:none;border-top:1px solid var(--rule-soft);
    border-bottom:1px solid var(--rule-soft);padding:14px 0;max-width:none}
  .big{font-size:42px}
  .reasons{max-width:none}
}
"""

    js = """
const btns=document.querySelectorAll('[data-filter]');
btns.forEach(b=>b.addEventListener('click',()=>{
  btns.forEach(x=>x.setAttribute('aria-pressed', x===b));
  const want=b.dataset.filter;
  document.querySelectorAll('.q tbody tr').forEach(tr=>{
    tr.hidden = !(want==='all' || tr.dataset.tier===want || tr.dataset.v===want);
  });
}));
"""

    def esc(x):
        return html.escape(str(x))

    qhtml = "\n".join(
        f'<tr data-tier="{r["tier"]}" data-v="{r["vclass"]}">'
        f'<td class="num">{r["rank"]}</td>'
        f'<td class="num">{r["acct"]}</td>'
        f'<td class="num">{r["score"]:.4f}</td>'
        f'<td><span class="tier {r["tier"]}">{esc(r["tier"])}</span><br>'
        f'<span style="color:var(--muted);font-size:13px">{esc(r["action"])}</span></td>'
        f'<td class="num">{r["cluster"] if r["cluster"] >= 0 else "-"}</td>'
        f'<td class="v {r["vclass"]}">{esc(r["verdict"])}</td>'
        f'<td class="reasons">{esc(r["reasons"])}</td></tr>'
        for r in rows)

    chtml = "\n".join(
        f'<tr><td><span class="tier {t}">{esc(t)}</span></td><td>{esc(act)}</td>'
        f'<td class="num">{acc:,}</td><td class="num">{m:,}</td>'
        f'<td class="num">{fp:,}</td><td class="num">{rupees(c)}</td>'
        f'<td class="num">{rupees(tot)}</td></tr>'
        for t, act, acc, m, fp, c, tot in cost_rows)

    rhtml = "\n".join(
        f'<tr><td class="num">{cid}</td><td class="num">{acc}</td>'
        f'<td class="num">{m}</td><td class="num">{pt}</td>'
        f'<td class="num">{fi}</td></tr>'
        for cid, acc, m, pt, fi in ring_rows) or \
        '<tr><td colspan="5" style="color:var(--muted)">No clusters of 4+</td></tr>'

    phtml = ", ".join(f"{esc(k)} ({v})" for k, v in fp_personas.items()) or "none"

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MuleGraph - review queue</title>
<style>{css}</style></head><body><div class="wrap">

<h1>Today's review queue</h1>
<p class="sub">Money mule detection across {n:,} accounts. The model ranks;
a separate policy decides what happens. Nothing here is a freeze decision.</p>

<div class="panel">
  <div class="split">
    <div><div class="big">{p_conf:.3f}</div>
      <div class="lbl">Precision measured against confirmed cases - what a
      fraud team can actually see</div></div>
    <div class="gap">{unconfirmed} of the accounts counted wrong here are real
      mules nobody ever confirmed</div>
    <div><div class="big">{p_true:.3f}</div>
      <div class="lbl">Precision measured against ground truth - what the model
      actually got right</div></div>
  </div>
  <div class="strip">
    <div class="stat"><b>{k80:,}</b><span>reviews to catch 80% of mules</span></div>
    <div class="stat"><b>{innocent}</b><span>genuinely legitimate accounts in that queue</span></div>
    <div class="stat"><b>{rupees(total_cost)}</b><span>cost of every mistake, all {n//1000}k accounts</span></div>
    <div class="stat"><b>{free_fp:,}</b><span>of {all_fp:,} mistakes cost the customer nothing</span></div>
  </div>
</div>

<h2>The queue</h2>
<p class="note">Ranked by score, with the reason each account surfaced. An
analyst cannot act on a bare number, and a customer deserves an explanation.</p>

<div class="filters">
  <button data-filter="all" aria-pressed="true">All</button>
  <button data-filter="critical" aria-pressed="false">Critical</button>
  <button data-filter="high" aria-pressed="false">High</button>
  <button data-filter="medium" aria-pressed="false">Medium</button>
  <button data-filter="miss" aria-pressed="false">False positives only</button>
</div>

<table class="q"><thead><tr>
<th class="num">#</th><th class="num">Account</th><th class="num">Score</th>
<th>Risk band and action</th><th class="num">Ring</th><th>Truth</th><th>Why it surfaced</th>
</tr></thead><tbody>
{qhtml}
</tbody></table>

<h2>What our mistakes cost</h2>
<p class="note">Because the action at low confidence is free, the system can
afford to look at far more accounts than a freeze-or-nothing system ever
could. The outbound-hold tier is the only one that does real harm, and it is
reserved for high confidence.</p>
<table><thead><tr>
<th>Risk band</th><th>Action</th><th class="num">Accounts</th>
<th class="num">Mules</th><th class="num">False positives</th>
<th class="num">Cost each</th><th class="num">Total</th>
</tr></thead><tbody>
{chtml}
</tbody></table>

<h2>Rings, not accounts</h2>
<p class="note">Telling an analyst that fourteen accounts are one operation is
more useful than fourteen separate alerts. It is also the difference between
blocking an operation and playing whack-a-mole while the operator activates
the next account of thirty.</p>
<table><thead><tr>
<th class="num">Cluster</th><th class="num">Accounts</th><th class="num">Mules</th>
<th class="num">Median pass-through</th><th class="num">Median fan-in</th>
</tr></thead><tbody>
{rhtml}
</tbody></table>

<h2>Who we got wrong</h2>
<p class="note">The people this model flags by mistake are a predictable set,
and worth naming: {phtml}. The cost of these errors falls hardest on people
least able to absorb it, which is the entire reason the response is graded
rather than binary.</p>

<p class="foot">Scores are out-of-fold predictions from 5-fold stratified
cross-validation - every account is scored by a model that never saw it.
40% of mule rings are deliberately left unlabelled, because no fraud team has
ever had a complete answer key. Synthetic data; the figures are an upper
bound, not a production estimate. Detection only.</p>

</div><script>{js}</script></body></html>
"""

    with open("dashboard.html", "w") as fh:
        fh.write(doc)
    print(f"wrote dashboard.html  ({len(doc)/1024:.0f} KB, {len(rows)} queue rows)")
    print(f"  precision {p_conf:.3f} confirmed / {p_true:.3f} truth")
    print(f"  {innocent} innocent accounts in a queue of {k80}")
    print("  open it with:  open dashboard.html")


if __name__ == "__main__":
    main()
