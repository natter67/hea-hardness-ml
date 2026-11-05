"""Same model, two ways of cross-validating it: random KFold vs GroupKFold.

Usage:
    python scripts/cv_split_demo.py                 # features A+B (default)
    python scripts/cv_split_demo.py --features B    # only the 4 physics features
    python scripts/cv_split_demo.py --decimals 2    # stricter composition grouping

THE POINT OF THIS SCRIPT
    Nothing about the model changes between the two runs. Only the *split*
    changes. Any difference in the scores is therefore entirely about how
    honestly each split measures "how well will this predict an alloy we
    haven't seen?" — which is what we actually care about.

    * KFold(shuffle=True): rows are dealt into folds at random. Because ~1/3 of
      rows share a composition with another row (see notebooks/01_eda.ipynb, Q4),
      the same alloy often sits in train *and* test. A tree ensemble then scores
      well partly by recognising the alloy, i.e. memorisation.

    * GroupKFold(groups=comp_key): every row of one normalized composition goes
      to the same fold, so the test fold contains only alloys the model has
      never seen. This mimics the real use case: proposing a *new* alloy.

    Expect GroupKFold to score worse. That is not the model getting worse; it
    is the estimate getting more honest.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
    No hyper-parameter tuning, no feature selection, no SHAP. Those come later,
    and when they do, they must happen *inside* each training fold (nested CV),
    or they leak test information just like duplicate compositions do.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_validate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make hea_ml importable
from hea_ml.data import DEFAULT_DECIMALS, load_hardness  # noqa: E402
from hea_ml.features import build_features  # noqa: E402

N_SPLITS = 5
SEED = 42  # fixed so that re-running gives identical numbers

# scikit-learn's convention is "higher score = better", so error metrics are
# exposed as *negative* MAE / RMSE. We flip the sign back when printing.
SCORING = {
    "R2": "r2",
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
}


def make_model() -> GradientBoostingRegressor:
    """One model definition, used for both CV schemes.

    GradientBoostingRegressor = an ensemble of shallow decision trees, each
    fitted to the residual errors of the ones before it.

    WHY gradient boosting as the demo model:
      * Trees don't care about feature scale, so we need no StandardScaler
        (one less thing that could leak if done outside the CV loop).
      * It handles the mix of 132 correlated Magpie columns + 4 physics
        columns without fuss.
      * It is *good at memorising*, which makes the KFold-vs-GroupKFold gap
        easy to see. A linear model would show a smaller gap.

    Settings are scikit-learn defaults except random_state (reproducibility)
    and subsample=0.8, which adds a little randomness per tree and is a
    standard, mild regulariser. NOT tuned — tuning is a later phase.
    """
    return GradientBoostingRegressor(random_state=SEED, subsample=0.8)


def summarise(name: str, cv_result: dict) -> dict[str, tuple[float, float]]:
    """Print mean ± std across folds for each metric and return them.

    WHY mean ± std and not just the mean: with ~500 rows, each test fold has
    ~100 rows and fold-to-fold variation is large. The std tells you whether
    a difference between two setups is bigger than that noise.
    (Strictly the folds aren't independent, so the std understates the true
    uncertainty a bit — fine for a comparison, not for a paper.)
    """
    out = {}
    parts = []
    for metric in SCORING:
        vals = cv_result[f"test_{metric}"]
        if metric != "R2":
            vals = -vals  # undo sklearn's negation
        out[metric] = (vals.mean(), vals.std())
        unit = "" if metric == "R2" else " HV"
        fmt = ".3f" if metric == "R2" else ".1f"
        parts.append(f"{metric} = {vals.mean():{fmt}} ± {vals.std():{fmt}}{unit}")
    print(f"{name:<34s} " + "   ".join(parts))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--features", choices=["A", "B", "AB"], default="AB",
                        help="A = Magpie (132), B = δ/ΔS/ΔH/VEC (4), AB = both")
    parser.add_argument("--decimals", type=int, default=DEFAULT_DECIMALS,
                        help="rounding of atomic fractions when defining composition groups")
    args = parser.parse_args()

    # ---- data --------------------------------------------------------------
    df = load_hardness(decimals=args.decimals)
    X = build_features(df["formula"], which=args.features)
    y = df["HV"].to_numpy()
    groups = df["comp_key"].to_numpy()
    print(f"{len(df)} rows, {len(np.unique(groups))} composition groups, "
          f"{X.shape[1]} features (set {args.features}); {N_SPLITS}-fold CV\n")

    # ---- the two splitters -------------------------------------------------
    # KFold must be shuffled: the CSV is ordered by paper, so unshuffled folds
    # would be "leave some papers out", a different (and also interesting)
    # experiment.
    random_cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    # GroupKFold balances fold sizes while keeping each group in one fold.
    # shuffle=True (sklearn >= 1.6) randomises which groups go together, so
    # results don't depend on the alphabetical order of comp_key.
    group_cv = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Sanity check that the grouped split really has no overlap — cheap
    # insurance against a bug that would make the whole comparison moot.
    for tr, te in group_cv.split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te]), "group leaked across folds!"

    # ---- evaluate ----------------------------------------------------------
    # cross_validate clones make_model() for every fold, so each fold starts
    # from an untrained model — no state carries over between folds.
    results = {}
    for name, cv, g in [
        ("Random KFold", random_cv, None),
        ("GroupKFold (by composition)", group_cv, groups),
    ]:
        res = cross_validate(make_model(), X, y, groups=g, cv=cv, scoring=SCORING, n_jobs=-1)
        results[name] = summarise(name, res)

    # ---- interpret ---------------------------------------------------------
    (r_mae, _), (g_mae, _) = results["Random KFold"]["MAE"], results["GroupKFold (by composition)"]["MAE"]
    (r_r2, _), (g_r2, _) = results["Random KFold"]["R2"], results["GroupKFold (by composition)"]["R2"]
    print(
        f"\nGrouping changes MAE by {g_mae - r_mae:+.1f} HV ({(g_mae / r_mae - 1):+.0%}) "
        f"and R² by {g_r2 - r_r2:+.3f}.\n"
        "The GroupKFold numbers are the ones to quote for 'predicting a new alloy'.\n"
        "For scale: the EDA's within-composition SD (~44 HV) is a rough floor for any\n"
        "composition-only model."
    )


if __name__ == "__main__":
    main()
