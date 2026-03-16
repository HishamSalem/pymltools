# predykt

A Python toolkit for rigorous feature interaction analysis in machine learning models. Combines cyclical optimal binning, SHAP interaction stability testing, and seed robustness validation into a unified workflow for credit risk and tabular ML.

---

## Why predykt?

Standard ML libraries treat feature analysis as a single-pass operation — fit once, read SHAP values, done. This works poorly when:

- Your temporal features are **cyclical** (hour-of-day, month-of-year) — standard binners don't know that 23:00 and 00:00 are adjacent
- Your SHAP interactions are **seed-dependent** — a strong-looking interaction from a single fit may vanish on the next random seed
- Your HPO result is **lucky** — the best config from your tuning run may only be best for that seed

predykt addresses each of these failure modes with dedicated, statistically grounded tools.

---

## Installation

```bash
pip install predykt
```

**Dependencies:** `numpy`, `numba`, `scikit-learn`, `pandas`, `shap`, `scipy`, `statsmodels`, `optbinning`, `matplotlib`, `seaborn`, `joblib`, `tqdm`

---

## Modules

| Module | What it does |
|--------|-------------|
| `CyclicalBinner` | IV-maximizing optimal binning for circular temporal features |
| `InteractionTester` | SHAP interaction stability testing via refit-across-seeds |
| `InteractionVoter` | Cross-algorithm voting to distinguish data interactions from algorithm artifacts |
| `SeedRobustnessValidator` | Statistical validation of hyperparameter config robustness across seeds |
| `FeatureBinningAnalyzer` | IV uplift screening for feature pair interactions via OptBinning |

---

## Quick Start

### 1. Cyclical Optimal Binning

Standard binners treat hour 23 and hour 0 as maximally distant. `CyclicalBinner` treats the domain as circular and finds the IV-maximizing partition accordingly.

```python
import numpy as np
from predykt import CyclicalBinner

# Simulate hour-of-day data with a fraud spike at night (22:00–02:00)
rng = np.random.default_rng(42)
n = 10_000
hours = rng.integers(0, 24, size=n)
# Higher fraud probability in late-night hours
fraud_prob = np.where((hours >= 22) | (hours <= 2), 0.15, 0.04)
y = rng.binomial(1, fraud_prob)

binner = CyclicalBinner(m=24, gamma=0.02, k_max=6)
binner.fit(hours, y)

print(f"Optimal bins: {binner.n_bins_}")
print(f"Split points: {binner.split_points_}")
print(f"IV: {binner.iv_:.4f}")
print(binner.result_.summary())
```

**Output (example):**
```
Optimal bins: 3
Split points: [ 3  8 22]
IV: 0.1823

   bin       range  count  count_%  events  non_events  event_rate       woe        iv
0    0    [3, 8)   2082    20.82      83        1999    0.039878 -0.3421    0.011
1    1   [8, 22)  5831    58.31     227        5604    0.038932 -0.3665    0.038
2    2  [22, 3)*  2087    20.87     312        1775    0.149496  1.1803    0.133
...
```

**Transform to WOE for scorecard:**
```python
# Bin index
binned = binner.transform(hours)

# WOE directly (for logistic regression scorecards)
woe_encoded = binner.transform_woe(hours)

# WOE lookup table for documentation
woe_table = binner.result_.woe_table()
```

---

### 2. SHAP Interaction Stability Testing

A SHAP interaction from a single model fit conflates a real data relationship with random seed luck. `InteractionTester` refits the model across N seeds and measures whether the interaction is a stable property.

```python
import pandas as pd
from xgboost import XGBClassifier
from predykt import InteractionTester

tester = InteractionTester(
    model_class=XGBClassifier,
    base_params={
        "n_estimators": 200,
        "max_depth": 5,
        "eval_metric": "logloss",
        "verbosity": 0,
    },
    seed_param="random_state",
    n_seeds=200,
    alpha=0.05,
    n_jobs=4,
)

# Step 1: cheap single-seed screen to identify candidate pairs
top_pairs = tester.get_top_n_interactions(X, y, n=10)

# Step 2: full stability test across 200 seeds
results = tester.test_pairs(X, y, top_pairs)

# Step 3: results with optional BH multiple testing correction
df = tester.results_to_dataframe(results, correction_method="fdr_bh")
print(df[["Feature_i", "Feature_j", "Instability_Score", "Per_Interaction_AUC", "Robust"]])
```

**What `instability_score` means:**
- Proportion of seeds where the interaction's sign opposes the majority direction (doubled for two-sidedness)
- Range [0, 1]. Lower = more stable.
- **This is not a frequentist p-value.** It measures algorithmic stability, not statistical significance under a null derived from the data-generating process.

```python
# Visualize seed distribution for a specific pair
tester.plot_interaction_distribution(results[0])

# Check if 200 seeds was enough for convergence
tester.plot_convergence(results[0])
```

---

### 3. Cross-Algorithm Voting

An interaction that is stable within XGBoost may be an artifact of gradient boosting's splitting strategy, not a property of the data. `InteractionVoter` runs the same stability test across multiple algorithm families and tallies votes.

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from predykt import InteractionVoter

configs = {
    "rf": {
        "model_class": RandomForestClassifier,
        "params": {"n_estimators": 200, "max_depth": 5, "n_jobs": -1},
        "seed_param": "random_state",
    },
    "xgb": {
        "model_class": XGBClassifier,
        "params": {"n_estimators": 200, "max_depth": 5, "eval_metric": "logloss", "verbosity": 0},
        "seed_param": "random_state",
    },
    "lgbm": {
        "model_class": LGBMClassifier,
        "params": {"n_estimators": 200, "max_depth": 5, "verbose": -1},
        "seed_param": "random_state",
    },
}

voter = InteractionVoter(configs, n_seeds=200, alpha=0.05, n_jobs=4)
vote_results = voter.vote(X, y, top_pairs)

# Summary table
summary = voter.summary(vote_results)
print(summary[["Feature_i", "Feature_j", "Votes", "Vote_Ratio", "Unanimous", "Mean_AUC"]])

# Heatmap: AUC by algorithm, * marks robust interactions
voter.plot_vote_heatmap(vote_results)
```

Unanimous interactions (all algorithms agree) are the most reliable candidates for feature engineering or regulatory documentation.

---

### 4. Seed Robustness Validation

HPO typically fixes a random seed during search, which means the "best" configuration may only be best for that initialization. `SeedRobustnessValidator` re-evaluates a fixed HP config across N seeds and runs formal statistical tests.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from predykt import SeedRobustnessValidator

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)

# The HP config you want to validate (found via HPO)
hp_config = {"n_estimators": 200, "max_depth": 6, "min_samples_split": 10}

def eval_fn(seed: int) -> float:
    clf = RandomForestClassifier(**hp_config, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    return roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

validator = SeedRobustnessValidator(
    eval_fn=eval_fn,
    n_seeds=100,
    metric_name="AUC",
    higher_is_better=True,
    sigma_max=0.005,  # domain-informed: 0.5% AUC std is acceptable for production
)

report = validator.run()
validator.print_report(report)
validator.plot_diagnostics(report)
```

**Statistical tests applied:**

| Test | Purpose |
|------|---------|
| Shapiro-Wilk | Gates parametric vs bootstrap path |
| Chi-square variance test (one-sided upper) | H₀: σ² ≤ σ²_max |
| 95/95 Tolerance interval | 95% confidence that 95% of future seed runs fall within [L, U] |
| Bootstrap CI for std | Non-parametric fallback when normality is violated |
| Coefficient of Variation | Relative dispersion summary |

**Verdict categories:** `ROBUST` / `MARGINAL` / `UNSTABLE`

> **Note on `sigma_max`:** If not set, defaults to 1% of the observed mean — a conservative auto-default. You should override this with a domain-informed threshold. In credit scoring, 0.5% AUC std (`sigma_max=0.005`) is a reasonable production stability requirement.

---

### 5. Feature Binning IV Uplift

Quick screening for feature pair interactions using OptBinning's 2D binning. The uplift heuristic (`IV_2D - (IV_1 + IV_2)`) identifies pairs where joint information exceeds the sum of marginal information — a signal worth investigating further.

```python
from predykt import FeatureBinningAnalyzer

analyzer = FeatureBinningAnalyzer(X, y)

feature_pairs = [
    ("age", "income"),
    ("loan_amount", "tenure"),
    ("utilization_rate", "delinquencies"),
]

results = analyzer.analyze_feature_combinations(feature_pairs)
print(analyzer.get_top_combinations())

# Inspect 2D binning table for a specific pair
table = analyzer.get_binning_details("age", "income")
print(table)
```

> **Interpretation note:** The IV uplift measure is a screening heuristic, not a formal interaction test. Pairs with high uplift are candidates for the more rigorous `InteractionTester` / `InteractionVoter` pipeline.

---

## Design Decisions

**Why refit across seeds instead of permuting on a fixed model?**
Permutation tests on a fixed model test whether the interaction is non-zero *for that fit*. Refitting tests whether the interaction is a stable property of the model family on this data — which is what matters for deployment. See the `InteractionTester` docstring for the full discussion.

**Why Numba for CyclicalBinner?**
Exhaustive enumeration of all k-partitions of a circular domain of cardinality m is O(C(m, k)) per k. For m=24, k=6, that's C(24,6) = 134,596 partitions. Numba JIT brings this from seconds to milliseconds.

**Why the 95/95 tolerance interval in SeedRobustnessValidator?**
A confidence interval on the mean tells you where the average seed lands. A tolerance interval tells you where *individual* seed runs land — which is what matters when you're deploying a model trained on a single seed. The 95/95 interval is the ISO 16269-6 standard for this use case.

---

## License

MIT

---

## Citation

If you use predykt in research or production systems, please cite the repository.
