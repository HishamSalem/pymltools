"""
Seed Robustness Validator
=========================
Framework-agnostic diagnostic tool that determines whether a hyperparameter
configuration is genuinely performant or an artifact of random seed luck.

Theory:
    HPO fixes a seed during search, conflating HP quality with initialization
    luck. This tool re-evaluates a fixed HP config across N seeds and applies
    statistical tests to quantify robustness.

Statistical Methods:
    1. Chi-square variance test (H0: sigma^2 <= sigma^2_max, one-sided upper)
    2. Normal tolerance interval (95/95: 95% confidence that 95% of future
       seed runs fall within [L, U])
    3. Coefficient of Variation (CV)
    4. Shapiro-Wilk normality test (gates parametric vs bootstrap path)
    5. Bootstrap CI for std when normality is violated

Usage:
    from seed_robustness_validator import SeedRobustnessValidator

    def my_eval_fn(seed: int) -> float:
        # Train model with given seed, return scalar metric
        model = MyModel(**my_hp_config, random_state=seed)
        model.fit(X_train, y_train)
        return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    validator = SeedRobustnessValidator(
        eval_fn=my_eval_fn,
        n_seeds=100,
        metric_name="AUC",
        higher_is_better=True,
    )
    report = validator.run()
    validator.print_report(report)
    validator.plot_diagnostics(report)
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Configuration & result containers
# ---------------------------------------------------------------------------

@dataclass
class RobustnessReport:
    """All diagnostic outputs from a single validation run."""

    # Raw data
    seeds: np.ndarray
    scores: np.ndarray
    metric_name: str
    higher_is_better: bool

    # Descriptive stats
    mean: float = 0.0
    std: float = 0.0
    cv: float = 0.0
    median: float = 0.0
    iqr: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    range_score: float = 0.0

    # Normality
    shapiro_stat: float = 0.0
    shapiro_p: float = 0.0
    is_normal: bool = False  # alpha=0.05

    # Chi-square variance test
    chisq_stat: float = 0.0
    chisq_p: float = 0.0
    sigma_max: float = 0.0
    variance_acceptable: bool = False

    # Tolerance interval (95/95)
    tol_lower: float = 0.0
    tol_upper: float = 0.0
    tol_k: float = 0.0

    # Bootstrap CI for std (used when non-normal)
    bootstrap_std_ci_lower: float = 0.0
    bootstrap_std_ci_upper: float = 0.0

    # Overall verdict
    verdict: str = ""
    confidence_notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

class SeedRobustnessValidator:
    """
    Evaluates a fixed hyperparameter config across many random seeds
    and runs statistical diagnostics to assess robustness.

    Parameters
    ----------
    eval_fn : Callable[[int], float]
        Function that accepts a seed (int) and returns a scalar performance
        metric. Must be deterministic for a given seed.
    n_seeds : int
        Number of seeds to evaluate. Recommend 100 for classical ML,
        30 minimum for stable std estimates.
    metric_name : str
        Display name for the metric (e.g., "AUC", "RMSE", "F1").
    higher_is_better : bool
        Whether higher metric values indicate better performance.
    sigma_max : float or None
        Maximum acceptable standard deviation. If None, auto-set to
        1% of the mean observed performance (a reasonable default for
        normalized metrics like AUC/F1). You should override this with
        a domain-informed value.
    seed_start : int
        Starting seed value. Seeds are generated as seed_start .. seed_start + n_seeds - 1.
    n_bootstrap : int
        Number of bootstrap resamples for non-parametric std CI.
    alpha : float
        Significance level for all tests.
    """

    def __init__(
        self,
        eval_fn: Callable[[int], float],
        n_seeds: int = 100,
        metric_name: str = "metric",
        higher_is_better: bool = True,
        sigma_max: Optional[float] = None,
        seed_start: int = 0,
        n_bootstrap: int = 10_000,
        alpha: float = 0.05,
    ):
        if n_seeds < 30:
            warnings.warn(
                f"n_seeds={n_seeds} is below 30. Std estimates will be unstable. "
                "Recommend >=30 (ideally 100).",
                UserWarning,
            )
        self.eval_fn = eval_fn
        self.n_seeds = n_seeds
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.sigma_max = sigma_max
        self.seed_start = seed_start
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha

    # -----------------------------------------------------------------------
    # Seed generation
    # -----------------------------------------------------------------------

    def _generate_seeds(self) -> np.ndarray:
        """Deterministic, well-spaced seeds derived from sequential indices."""
        seeds = []
        for i in range(self.n_seeds):
            # Hash-based seed derivation avoids correlated RNG streams
            h = hashlib.sha256(f"robustness_seed_{self.seed_start + i}".encode())
            seed_val = int(h.hexdigest()[:8], 16) % (2**31)
            seeds.append(seed_val)
        return np.array(seeds, dtype=np.int64)

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def _evaluate_seeds(self, seeds: np.ndarray) -> np.ndarray:
        """Run eval_fn across all seeds. Fails fast on non-finite returns."""
        scores = np.empty(len(seeds), dtype=np.float64)
        for i, seed in enumerate(seeds):
            score = self.eval_fn(int(seed))
            if not np.isfinite(score):
                raise ValueError(
                    f"eval_fn returned non-finite value ({score}) for seed={seed}. "
                    "Check your training pipeline."
                )
            scores[i] = score
            if (i + 1) % max(1, len(seeds) // 10) == 0:
                print(f"  [{i + 1}/{len(seeds)}] seeds evaluated...")
        return scores

    # -----------------------------------------------------------------------
    # Statistical tests
    # -----------------------------------------------------------------------

    @staticmethod
    def _shapiro_wilk(scores: np.ndarray, alpha: float) -> tuple[float, float, bool]:
        """Test normality. Required gate for parametric chi-square test."""
        stat, p = stats.shapiro(scores)
        return stat, p, p > alpha

    @staticmethod
    def _chi2_variance_test(
        scores: np.ndarray, sigma_max: float, alpha: float
    ) -> tuple[float, float, bool]:
        """
        One-sided upper chi-square test.
        H0: sigma^2 <= sigma_max^2
        H1: sigma^2 > sigma_max^2

        Reject H0 (variance unacceptable) if p < alpha.
        """
        n = len(scores)
        s2 = np.var(scores, ddof=1)
        chi2_stat = (n - 1) * s2 / (sigma_max**2)
        # Upper tail: P(X > chi2_stat) where X ~ chi2(n-1)
        p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=n - 1)
        # Acceptable = fail to reject H0
        acceptable = p_value >= alpha
        return chi2_stat, p_value, acceptable

    @staticmethod
    def _tolerance_interval_95_95(scores: np.ndarray) -> tuple[float, float, float]:
        """
        Normal-theory two-sided tolerance interval.
        95% confidence that 95% of future observations lie in [L, U].

        Uses the k-factor from Howe (1969) / ISO 16269-6 approximation.
        """
        n = len(scores)
        p = 0.95  # coverage
        gamma = 0.95  # confidence
        xbar = np.mean(scores)
        s = np.std(scores, ddof=1)

        # k-factor: one-sided normal tolerance factor, converted to two-sided
        # Exact: k such that P(covers p fraction | n) = gamma
        # Approximation via scipy (non-central t approach)
        z_p = stats.norm.ppf((1 + p) / 2)
        chi2_val = stats.chi2.ppf(1 - gamma, df=n - 1)
        k = z_p * np.sqrt((n - 1) / chi2_val) * np.sqrt(1 + 1 / n)

        lower = xbar - k * s
        upper = xbar + k * s
        return lower, upper, k

    @staticmethod
    def _bootstrap_std_ci(
        scores: np.ndarray, n_bootstrap: int, alpha: float
    ) -> tuple[float, float]:
        """Non-parametric bootstrap CI for standard deviation."""
        rng = np.random.default_rng(42)
        boot_stds = np.empty(n_bootstrap)
        n = len(scores)
        for i in range(n_bootstrap):
            sample = rng.choice(scores, size=n, replace=True)
            boot_stds[i] = np.std(sample, ddof=1)
        lower = np.percentile(boot_stds, 100 * alpha / 2)
        upper = np.percentile(boot_stds, 100 * (1 - alpha / 2))
        return lower, upper

    # -----------------------------------------------------------------------
    # Verdict logic
    # -----------------------------------------------------------------------

    def _compute_verdict(self, report: RobustnessReport) -> None:
        """Synthesize all test results into a single verdict + notes."""
        notes = []

        # 1. Normality check
        if not report.is_normal:
            notes.append(
                f"Shapiro-Wilk rejected normality (p={report.shapiro_p:.4f}). "
                "Chi-square and tolerance interval assume normality; interpret "
                "with caution. Bootstrap CI for std is more reliable here."
            )

        # 2. Variance test
        if report.variance_acceptable:
            notes.append(
                f"Chi-square: PASS. Cannot reject H0 (sigma <= {report.sigma_max:.6f}) "
                f"at alpha={self.alpha}. p={report.chisq_p:.4f}."
            )
        else:
            notes.append(
                f"Chi-square: FAIL. Reject H0; observed std ({report.std:.6f}) "
                f"exceeds sigma_max ({report.sigma_max:.6f}) with statistical "
                f"significance. p={report.chisq_p:.4f}."
            )

        # 3. Tolerance interval assessment
        tol_range = report.tol_upper - report.tol_lower
        notes.append(
            f"95/95 Tolerance interval: [{report.tol_lower:.6f}, {report.tol_upper:.6f}] "
            f"(width={tol_range:.6f}). This is the range you should expect 95% of "
            f"future seed runs to fall within."
        )

        # 4. CV assessment
        cv_pct = report.cv * 100
        if cv_pct < 1.0:
            cv_verdict = "excellent (CV < 1%)"
        elif cv_pct < 3.0:
            cv_verdict = "acceptable (1% <= CV < 3%)"
        elif cv_pct < 5.0:
            cv_verdict = "marginal (3% <= CV < 5%), investigate"
        else:
            cv_verdict = "poor (CV >= 5%), this config is seed-sensitive"
        notes.append(f"Coefficient of Variation: {cv_pct:.2f}% - {cv_verdict}.")

        # 5. Overall
        failures = []
        if not report.variance_acceptable:
            failures.append("chi-square variance test")
        if cv_pct >= 5.0:
            failures.append("CV threshold")

        if not failures:
            report.verdict = "ROBUST"
        elif len(failures) == 1:
            report.verdict = "MARGINAL"
        else:
            report.verdict = "UNSTABLE"

        report.confidence_notes = notes

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def run(self) -> RobustnessReport:
        """Execute full seed robustness diagnostic. Returns RobustnessReport."""
        print(f"=== Seed Robustness Validation ({self.n_seeds} seeds) ===")
        print(f"Metric: {self.metric_name} ({'higher' if self.higher_is_better else 'lower'} is better)\n")

        # Generate & evaluate
        seeds = self._generate_seeds()
        print("Evaluating seeds...")
        scores = self._evaluate_seeds(seeds)
        print()

        # Descriptive stats
        mean = np.mean(scores)
        std_val = np.std(scores, ddof=1)
        cv = std_val / abs(mean) if abs(mean) > 1e-12 else float("inf")

        # Auto-set sigma_max if not provided
        sigma_max = self.sigma_max
        if sigma_max is None:
            sigma_max = 0.01 * abs(mean)
            print(
                f"sigma_max not set. Auto-defaulting to 1% of mean: {sigma_max:.6f}. "
                "Override with a domain-informed value.\n"
            )

        # Build report
        report = RobustnessReport(
            seeds=seeds,
            scores=scores,
            metric_name=self.metric_name,
            higher_is_better=self.higher_is_better,
            mean=mean,
            std=std_val,
            cv=cv,
            median=float(np.median(scores)),
            iqr=float(np.percentile(scores, 75) - np.percentile(scores, 25)),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            range_score=float(np.max(scores) - np.min(scores)),
            sigma_max=sigma_max,
        )

        # Statistical tests
        report.shapiro_stat, report.shapiro_p, report.is_normal = self._shapiro_wilk(
            scores, self.alpha
        )
        report.chisq_stat, report.chisq_p, report.variance_acceptable = (
            self._chi2_variance_test(scores, sigma_max, self.alpha)
        )
        report.tol_lower, report.tol_upper, report.tol_k = (
            self._tolerance_interval_95_95(scores)
        )
        report.bootstrap_std_ci_lower, report.bootstrap_std_ci_upper = (
            self._bootstrap_std_ci(scores, self.n_bootstrap, self.alpha)
        )

        # Verdict
        self._compute_verdict(report)

        return report

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def print_report(self, report: RobustnessReport) -> None:
        """Print human-readable diagnostic report to stdout."""
        sep = "=" * 65
        print(sep)
        print(f"  SEED ROBUSTNESS REPORT: {report.metric_name}")
        print(sep)

        print(f"\n{'Direction:':<30} {'higher is better' if report.higher_is_better else 'lower is better'}")
        print(f"{'Seeds evaluated:':<30} {len(report.seeds)}")
        print(f"{'sigma_max (threshold):':<30} {report.sigma_max:.6f}")

        print(f"\n--- Descriptive Statistics ---")
        print(f"{'Mean:':<30} {report.mean:.6f}")
        print(f"{'Std (sample):':<30} {report.std:.6f}")
        print(f"{'CV:':<30} {report.cv * 100:.2f}%")
        print(f"{'Median:':<30} {report.median:.6f}")
        print(f"{'IQR:':<30} {report.iqr:.6f}")
        print(f"{'Min:':<30} {report.min_score:.6f}")
        print(f"{'Max:':<30} {report.max_score:.6f}")
        print(f"{'Range:':<30} {report.range_score:.6f}")

        print(f"\n--- Normality (Shapiro-Wilk) ---")
        print(f"{'Statistic:':<30} {report.shapiro_stat:.6f}")
        print(f"{'p-value:':<30} {report.shapiro_p:.4f}")
        print(f"{'Normal (alpha=' + str(self.alpha) + '):':<30} {'YES' if report.is_normal else 'NO'}")

        print(f"\n--- Chi-Square Variance Test ---")
        print(f"{'H0: sigma^2 <= sigma_max^2'}")
        print(f"{'Test statistic:':<30} {report.chisq_stat:.4f}")
        print(f"{'p-value:':<30} {report.chisq_p:.4f}")
        print(f"{'Variance acceptable:':<30} {'YES' if report.variance_acceptable else 'NO'}")

        print(f"\n--- 95/95 Tolerance Interval ---")
        print(f"{'k-factor:':<30} {report.tol_k:.4f}")
        print(f"{'Lower bound:':<30} {report.tol_lower:.6f}")
        print(f"{'Upper bound:':<30} {report.tol_upper:.6f}")
        print(f"{'Width:':<30} {report.tol_upper - report.tol_lower:.6f}")

        print(f"\n--- Bootstrap 95% CI for Std ---")
        print(f"{'Lower:':<30} {report.bootstrap_std_ci_lower:.6f}")
        print(f"{'Upper:':<30} {report.bootstrap_std_ci_upper:.6f}")

        print(f"\n--- VERDICT: {report.verdict} ---")
        for i, note in enumerate(report.confidence_notes, 1):
            print(f"  {i}. {note}")

        print(sep)

    def plot_diagnostics(
        self, report: RobustnessReport, save_path: Optional[str] = None
    ) -> None:
        """
        Generate 4-panel diagnostic plot:
          1. Histogram + KDE of scores with tolerance interval
          2. Ordered seed performance (index plot)
          3. Q-Q plot (normality visual)
          4. Bootstrap distribution of std with CI

        Parameters
        ----------
        save_path : str or None
            If provided, saves the figure to this path. Otherwise calls plt.show().
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Seed Robustness Diagnostics: {report.metric_name}  "
            f"[Verdict: {report.verdict}]",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        scores = report.scores

        # --- Panel 1: Histogram + tolerance interval ---
        ax1 = axes[0, 0]
        ax1.hist(scores, bins="auto", density=True, alpha=0.6, color="#4C72B0",
                 edgecolor="white", label="Observed")
        # KDE
        if report.std > 1e-12:
            xs = np.linspace(scores.min() - 2 * report.std,
                             scores.max() + 2 * report.std, 300)
            kde = stats.gaussian_kde(scores)
            ax1.plot(xs, kde(xs), color="#C44E52", lw=2, label="KDE")
        ax1.axvline(report.mean, color="black", ls="--", lw=1.5, label=f"Mean={report.mean:.4f}")
        ax1.axvspan(report.tol_lower, report.tol_upper, alpha=0.12, color="green",
                    label=f"95/95 TI [{report.tol_lower:.4f}, {report.tol_upper:.4f}]")
        ax1.set_xlabel(report.metric_name)
        ax1.set_ylabel("Density")
        ax1.set_title("Score Distribution + Tolerance Interval")
        ax1.legend(fontsize=8)

        # --- Panel 2: Index plot (ordered by evaluation) ---
        ax2 = axes[0, 1]
        ax2.scatter(range(len(scores)), scores, s=12, alpha=0.6, color="#4C72B0")
        ax2.axhline(report.mean, color="black", ls="--", lw=1, label="Mean")
        ax2.axhline(report.mean + 2 * report.std, color="red", ls=":", lw=1, label="±2σ")
        ax2.axhline(report.mean - 2 * report.std, color="red", ls=":", lw=1)
        ax2.set_xlabel("Seed index")
        ax2.set_ylabel(report.metric_name)
        ax2.set_title("Per-Seed Performance")
        ax2.legend(fontsize=8)

        # --- Panel 3: Q-Q plot ---
        ax3 = axes[1, 0]
        stats.probplot(scores, dist="norm", plot=ax3)
        ax3.set_title(f"Q-Q Plot (Shapiro p={report.shapiro_p:.4f})")
        ax3.get_lines()[0].set_markersize(4)

        # --- Panel 4: Bootstrap std distribution ---
        ax4 = axes[1, 1]
        rng = np.random.default_rng(42)
        boot_stds = np.array([
            np.std(rng.choice(scores, size=len(scores), replace=True), ddof=1)
            for _ in range(self.n_bootstrap)
        ])
        ax4.hist(boot_stds, bins="auto", density=True, alpha=0.6, color="#DD8452",
                 edgecolor="white")
        ax4.axvline(report.std, color="black", ls="--", lw=1.5, label=f"Observed std={report.std:.5f}")
        ax4.axvline(report.bootstrap_std_ci_lower, color="green", ls=":", lw=1.5)
        ax4.axvline(report.bootstrap_std_ci_upper, color="green", ls=":", lw=1.5,
                    label=f"95% CI [{report.bootstrap_std_ci_lower:.5f}, {report.bootstrap_std_ci_upper:.5f}]")
        if report.sigma_max > 0:
            ax4.axvline(report.sigma_max, color="red", ls="-", lw=2,
                        label=f"σ_max={report.sigma_max:.5f}")
        ax4.set_xlabel("Standard Deviation")
        ax4.set_ylabel("Density")
        ax4.set_title("Bootstrap Distribution of Std")
        ax4.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Diagnostic plot saved to: {save_path}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     from sklearn.datasets import make_classification
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.metrics import roc_auc_score
#     from sklearn.model_selection import train_test_split

#     # Fixed dataset (seed for data only , held constant)
#     X, y = make_classification(
#         n_samples=1000, n_features=20, n_informative=10,
#         n_redundant=5, random_state=999, flip_y=0.05,
#     )
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=999, stratify=y,
#     )

#     # The HP config under test (fixed)
#     hp_config = dict(n_estimators=100, max_depth=5, min_samples_split=5)

#     def eval_fn(seed: int) -> float:
#         """Train RF with fixed HPs, variable seed. Return AUC."""
#         clf = RandomForestClassifier(**hp_config, random_state=seed, n_jobs=-1)
#         clf.fit(X_train, y_train)
#         proba = clf.predict_proba(X_test)[:, 1]
#         return roc_auc_score(y_test, proba)

#     validator = SeedRobustnessValidator(
#         eval_fn=eval_fn,
#         n_seeds=100,
#         metric_name="AUC",
#         higher_is_better=True,
#         sigma_max=0.005,  # domain-informed: 0.5% AUC std is acceptable
#     )

#     report = validator.run()
#     validator.print_report(report)
#     validator.plot_diagnostics(report, save_path="seed_robustness_diagnostics.png")
