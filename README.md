# predykt

A Python toolkit for rigorous feature interaction analysis in machine learning models. Combines cyclical optimal binning, SHAP interaction stability testing, FWL representation testing, and seed robustness validation into a unified workflow for credit risk and tabular ML.

## Why predykt?

Standard ML libraries treat feature analysis as a single-pass operation: fit once, read SHAP values, done. This works poorly when:

- Your temporal features are **cyclical** (hour-of-day, month-of-year): standard binners don't know that 23:00 and 00:00 are adjacent
- Your SHAP interactions are **seed-dependent**: a strong-looking interaction from a single fit may vanish on the next random seed
- Your HPO result is **lucky**: the best config from your tuning run may only be best for that seed
- Your engineered feature needs **validation**: a candidate transformation may not explain residual structure the base model missed

predykt addresses each of these failure modes with dedicated, statistically grounded tools.

## Installation

```bash
pip install predykt
```

**Dependencies:** `numpy`, `numba`, `scikit-learn`, `pandas`, `shap`, `scipy`, `statsmodels`, `optbinning`, `matplotlib`, `seaborn`, `joblib`, `tqdm`

## Modules

| Module | What it does |
|--------|-------------|
| `CyclicalBinner` | IV-maximizing optimal binning for circular temporal features |
| `InteractionTester` | SHAP interaction stability testing via refit-across-seeds |
| `InteractionVoter` | Cross-algorithm voting to distinguish data interactions from algorithm artifacts |
| `SeedRobustnessValidator` | Statistical validation of hyperparameter config robustness across seeds |
| `FeatureBinningAnalyzer` | IV uplift screening for feature pair interactions via OptBinning |
| `FWLRepresentationTester` | FWL-based test of whether an engineered representation explains base-model residuals |
| `SHAPInteractionAnalyzer` | Three-layer SHAP attribution corrected for collinearity and cross-group aliasing |

## Quick Start

### 1. Cyclical Optimal Binning

Standard binners treat hour 23 and hour 0 as maximally distant. `CyclicalBinner` treats the domain as circular and finds the IV-maximizing partition accordingly.

```python
import numpy as np
from predykt import CyclicalBinner

# Simulate hour-of-day data with a fraud spike at night (22:00-02:00)
rng = np.random.default_rng(42)
n = 10_000
hours = rng.integers(0, 24, size=n)
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
| Chi-square variance test (one-sided upper) | H0: sigma^2 <= sigma^2_max |
| 95/95 Tolerance interval | 95% confidence that 95% of future seed runs fall within [L, U] |
| Bootstrap CI for std | Non-parametric fallback when normality is violated |
| Coefficient of Variation | Relative dispersion summary |

**Verdict categories:** `ROBUST` / `MARGINAL` / `UNSTABLE`

> **Note on `sigma_max`:** If not set, defaults to 1% of the observed mean, a conservative auto-default. You should override this with a domain-informed threshold. In credit scoring, 0.5% AUC std (`sigma_max=0.005`) is a reasonable production stability requirement.

### 5. Feature Binning IV Uplift

Quick screening for feature pair interactions using OptBinning's 2D binning. The uplift heuristic (`IV_2D - (IV_1 + IV_2)`) identifies pairs where joint information exceeds the sum of marginal information, a signal worth investigating further.

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

### 6. FWL Representation Testing

After confirming an interaction pair is stable, `FWLRepresentationTester` answers: does a specific engineered transformation of that pair explain structure the base model missed?

The test uses the Frisch-Waugh-Lovell theorem. Stage 1 computes out-of-fold residuals Ỹ = y - p̂ via K-fold cross-fitting. Stage 2 regresses Ỹ on the candidate feature Tk and tests H0: beta1 = 0. A significant result means Tk captures signal the base model failed to learn.

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from predykt import FWLRepresentationTester, OLSEstimator, HSICEstimator

# Candidate representations of the (age, income) interaction
reps = pd.DataFrame({
    "product":   X["age"] * X["income"],
    "ratio":     X["age"] / (X["income"] + 1),
    "log_ratio": np.log1p(X["age"]) - np.log1p(X["income"]),
})

tester = FWLRepresentationTester(
    model=GradientBoostingClassifier(n_estimators=200, random_state=42),
    criterion=[OLSEstimator(), HSICEstimator(n_permutations=500)],
    n_folds=5,
    alpha=0.05,
)

tester.fit(
    feature_pairs=[("age", "income")],
    X=X,
    y=y,
    representations=reps,
)

# Summary table: beta, t-stat, p-value, BH-corrected p-value, winner flag
print(tester.results_to_dataframe())

# Best representation per pair
winners = tester.winning_representations()

# Add placebo and bootstrap refutation checks
tester.refute(n_permutations=100, n_bootstrap=50)
print(tester.results_to_dataframe()[["representation", "rejected", "robust"]])
```

**Criteria:**

| Criterion | What it tests |
|-----------|--------------|
| `OLSEstimator` | Linear association (HC3 robust SE, handles heteroskedastic residuals) |
| `HSICEstimator` | Nonlinear / non-monotone dependence (kernel-based, permutation p-value) |
| `CustomEstimator` | Any user-supplied callable returning a `Stage2Result` |

**Precomputed residuals (Mode B):** If you already have OOF residuals from a prior run, pass `Y_resid=` directly to skip the Stage 1 cross-fitting.

```python
tester.fit(
    feature_pairs=[("age", "income")],
    X=X, y=y,
    representations=reps,
    Y_resid=precomputed_residuals,
)
```

### 7. SHAP Interaction Analyzer

When a model contains both raw features and engineered interactions, raw SHAP values are aliased by collinearity. `SHAPInteractionAnalyzer` provides three progressively purer attribution layers.

```python
from predykt import SHAPInteractionAnalyzer

groups = {
    "demographics": ["age", "income"],
    "credit":       ["utilization_rate", "delinquencies", "loan_amount"],
    "temporal":     ["tenure", "hour_bin", "month"],
}

analyzer = SHAPInteractionAnalyzer(interaction_groups=groups, layers=[1, 2, 3])
analyzer.fit(model=fitted_model, X=X_test)

# Layer 1: group total SHAP (sum within group)
l1 = analyzer.layer_1_group_total()

# Layer 2: net group effects (Layer 1 minus cross-group interaction contributions)
l2 = analyzer.layer_2_net_group_effects()

# Layer 3: pure main effects per feature (diagonal of shap_interaction_values)
l3 = analyzer.layer_3_pure_main_effects()

# Global importance summary for each layer
print(analyzer.summary(layer=1))
print(analyzer.summary(layer=2))

# Side-by-side comparison: layer_1 vs layer_2 per group
group_comparison, feature_effects = analyzer.compare_layers()
print(group_comparison)
```

**Reading the layers:**

| Comparison | What it tells you |
|------------|------------------|
| Layer 1 - Layer 2 per group | How much of the group's apparent importance comes from cross-group interactions |
| Layer 2 - sum(Layer 3 within group) | Within-group collinearity aliasing even after cross-group correction |

## Design Decisions

**Why refit across seeds instead of permuting on a fixed model?**
Permutation tests on a fixed model test whether the interaction is non-zero for that fit. Refitting tests whether the interaction is a stable property of the model family on this data, which is what matters for deployment. See the `InteractionTester` docstring for the full discussion.

**Why Numba for CyclicalBinner?**
Exhaustive enumeration of all k-partitions of a circular domain of cardinality m is O(C(m, k)) per k. For m=24, k=6, that's C(24,6) = 134,596 partitions. Numba JIT brings this from seconds to milliseconds.

**Why the 95/95 tolerance interval in SeedRobustnessValidator?**
A confidence interval on the mean tells you where the average seed lands. A tolerance interval tells you where individual seed runs land, which is what matters when you're deploying a model trained on a single seed. The 95/95 interval is the ISO 16269-6 standard for this use case.

**Why HC3 robust standard errors in OLSEstimator?**
For binary targets, residuals Ỹ = y - p̂ have observation-specific variance p̂(1-p̂). OLS with homoskedastic standard errors is misspecified. HC3 (MacKinnon & White 1985) corrects this and is the mandatory default.

**Why HSIC alongside OLS?**
OLS only detects linear association. HSIC (Hilbert-Schmidt Independence Criterion) is a kernel-based nonparametric test that detects any dependence structure, including nonlinear and non-monotone relationships. Running both gives a more complete picture of whether a representation carries signal.

## License

MIT

## Citation

If you use predykt in research or production systems, please cite the repository.
