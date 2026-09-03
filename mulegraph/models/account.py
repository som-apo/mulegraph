"""Account-level scoring model and the ablation.

Gradient boosting, because the core hypothesis is that no single signal
separates mules -- a careful mule keeps every individual feature inside
the acceptable range. What is rare is the *combination*. A tree ensemble
finds those interaction pockets automatically; a threshold rulebook
cannot.

We use sklearn's HistGradientBoostingClassifier rather than XGBoost: same
algorithm family, no OpenMP dependency, one less thing to install.

THE ABLATION
------------
Config A  raw week-9 behaviour only
Config B  + the account's own prior 8 weeks
Config C  + graph structure

Reported per difficulty. If the graph layer does not lift subtle recall,
that is a finding and we report it as one -- the table exists to test the
claim, not to decorate it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from ..features.account import GROUP_RAW, GROUP_BASELINE, load_features
from ..eval.metrics import evaluate, print_report, ablation_table


def cv_scores(X: pd.DataFrame, y: np.ndarray, seed: int = 0,
              n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predictions so every account is scored by a model that
    never saw it.

    Stratified because at 0.5% prevalence an unstratified split can hand a
    fold almost no positives.
    """
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        )
        clf.fit(X.iloc[tr], y[tr])
        oof[te] = clf.predict_proba(X.iloc[te])[:, 1]
    return oof


def feature_importance(feats: pd.DataFrame, cols, seed: int = 0,
                       n_repeats: int = 3) -> pd.DataFrame:
    """Permutation importance -- what the model actually leans on.

    Worth checking against intuition: if one feature dominates completely,
    that is usually a leak rather than a discovery. That check is what
    caught an early bug where only mule transfers were rounded to clean
    figures, handing the model a free answer key.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score, make_scorer

    y = feats["is_mule"].astype(int).to_numpy()
    X = feats[[c for c in cols if c in feats.columns]]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                         random_state=seed)
    clf.fit(X, y)
    scorer = make_scorer(average_precision_score, response_method="predict_proba")
    r = permutation_importance(clf, X, y, scoring=scorer,
                               n_repeats=n_repeats, random_state=seed)
    return (pd.DataFrame({"feature": X.columns, "importance": r.importances_mean})
            .sort_values("importance", ascending=False, ignore_index=True))


if __name__ == "__main__":
    feats = load_features()
    budget = 200
    y = feats["is_mule"].astype(int).to_numpy()
    d = feats["mule_difficulty"].astype(str).to_numpy()
    print(f"population {len(feats):,}  mules {y.sum()}  budget {budget}")

    results = []
    for label, cols in [("A: raw behaviour", GROUP_RAW),
                        ("B: + self-baseline", GROUP_RAW + GROUP_BASELINE)]:
        cols = [c for c in cols if c in feats.columns]
        oof = cv_scores(feats[cols], y)
        r = evaluate(y, oof, d, budget=budget, label=label)
        results.append(r)
        print_report(r, budget=budget)

    print("\n=== ABLATION (account level only) ===")
    print(ablation_table(results, budget=budget).to_string(index=False))
