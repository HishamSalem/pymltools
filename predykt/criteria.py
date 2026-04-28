"""
Stage-2 estimators for ResidualRepresentationTester.

In the residual framework the stage-2 question is:

    Does Tₖ explain the base-model residuals Ỹ?

This is operationalised as regressing Ỹ on Tₖ and testing H₀: β₁ = 0.
The abstract base class standardises the interface so ResidualRepresentationTester
is estimator-agnostic. Users can subclass to use any test they like.

Default: OLSEstimator with HC3 heteroskedasticity-robust standard errors.
HC3 is mandatory (not optional) for binary classification residuals because
Var(Ỹᵢ) = p̂ᵢ(1 − p̂ᵢ), observation-specific, never constant.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import statsmodels.api as sm


# =============================================================================
# KERNEL HELPERS (used by HSICEstimator)
# =============================================================================

def _rbf_kernel(x: np.ndarray, bandwidth: Optional[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    dists_sq = (x[:, None] - x[None, :]) ** 2
    if bandwidth is None:
        nonzero_sq = dists_sq[dists_sq > 0]
        bandwidth = float(np.sqrt(np.median(nonzero_sq))) if nonzero_sq.size else 1.0
    return np.exp(-dists_sq / (2.0 * bandwidth ** 2))


def _hsic_statistic(K: np.ndarray, L: np.ndarray) -> float:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    return float(np.trace(Kc @ Lc) / (n - 1) ** 2)


# =============================================================================
# RESULT CONTAINER
# =============================================================================

@dataclass
class Stage2Result:
    """
    Unified result from a stage-2 residual regression.

    Attributes
    ----------
    beta : float
        Estimated coefficient on Tₖ in Ỹ ~ β₀ + β₁·Tₖ.
    t_stat : float
        t-statistic for β₁ (signed).
    pvalue : float
        Two-sided p-value for H₀: β₁ = 0.
    significant : bool
        True if pvalue < alpha.
    method : str
        Estimator identifier, e.g. "ols_hc3".
    model_result : Any
        Full underlying model result object (e.g. statsmodels RegressionResults).
        Call .summary() on this for the complete diagnostic output.
    """
    beta: float
    t_stat: float
    pvalue: float
    significant: bool
    method: str
    model_result: Any = field(repr=False)


# =============================================================================
# ABSTRACT BASE
# =============================================================================

class Stage2Estimator(ABC):
    """
    Abstract base for stage-2 estimators in the residual framework.

    Subclass this to plug any regression or dependence test into
    ResidualRepresentationTester. The only requirement is that fit() returns
    a Stage2Result.

    Example: custom Spearman rank criterion:

        class SpearmanEstimator(Stage2Estimator):
            def __init__(self, alpha=0.05):
                self.alpha = alpha
            def fit(self, T_k, Y_resid):
                from scipy.stats import spearmanr
                rho, pval = spearmanr(T_k, Y_resid)
                return Stage2Result(
                    beta=rho, t_stat=rho, pvalue=pval,
                    significant=pval < self.alpha,
                    method='spearman', model_result=None,
                )
    """

    @abstractmethod
    def fit(self, T_k: np.ndarray, Y_resid: np.ndarray) -> Stage2Result:
        """
        Fit the stage-2 model.

        Parameters
        ----------
        T_k : np.ndarray, shape (n,)
            Candidate representation, engineered by the user. Passed as-is;
            no transforms are applied here.
        Y_resid : np.ndarray, shape (n,)
            OOF residuals Ỹ = y − p̂ from the base model.

        Returns
        -------
        Stage2Result
        """


# =============================================================================
# OLS WITH ROBUST STANDARD ERRORS (default)
# =============================================================================

class OLSEstimator(Stage2Estimator):
    """
    Stage-2 OLS with heteroskedasticity-robust standard errors.

    Fits Ỹ = β₀ + β₁·Tₖ + ε using statsmodels OLS, then tests H₀: β₁ = 0
    using the robust t-statistic.

    Parameters
    ----------
    cov_type : str, default="HC3"
        Covariance estimator passed to statsmodels .fit(cov_type=...).
        "HC3" (MacKinnon & White 1985) is the recommended choice for
        binary classification residuals; it is the standard small-sample
        heteroskedasticity-robust estimator.
        Other valid options: "HC0", "HC1", "HC2", "HAC".
    alpha : float, default=0.05
        Significance threshold for the significant flag in Stage2Result.

    Notes
    -----
    WHY HC3 IS NOT OPTIONAL:
        For binary targets, residuals are Ỹᵢ = yᵢ − p̂ᵢ where yᵢ ∈ {0,1}.
        Var(Ỹᵢ) = p̂ᵢ(1 − p̂ᵢ), observation-specific, never constant.
        OLS with homoskedastic SEs is misspecified. HC3 corrects this.

    The full statsmodels result is available on Stage2Result.model_result,
    so you can call result.model_result.summary() for the complete output.
    """

    def __init__(self, cov_type: str = "HC3", alpha: float = 0.05):
        self.cov_type = cov_type
        self.alpha = alpha

    def fit(self, T_k: np.ndarray, Y_resid: np.ndarray) -> Stage2Result:
        T_k = np.asarray(T_k, dtype=float)
        Y_resid = np.asarray(Y_resid, dtype=float)

        X_design = sm.add_constant(T_k, has_constant="add")
        ols_result = sm.OLS(Y_resid, X_design).fit(cov_type=self.cov_type)

        beta = float(ols_result.params[1])
        t_stat = float(ols_result.tvalues[1])
        pvalue = float(ols_result.pvalues[1])

        return Stage2Result(
            beta=beta,
            t_stat=t_stat,
            pvalue=pvalue,
            significant=pvalue < self.alpha,
            method=f"ols_{self.cov_type.lower()}",
            model_result=ols_result,
        )


# =============================================================================
# CUSTOM (user-supplied callable)
# =============================================================================

class CustomEstimator(Stage2Estimator):
    """
    Wraps a user-supplied callable as a Stage2Estimator.

    Parameters
    ----------
    fn : Callable[[np.ndarray, np.ndarray], Stage2Result]
        Signature: fn(T_k, Y_resid) -> Stage2Result.
    """

    def __init__(self, fn: Callable):
        self.fn = fn

    def fit(self, T_k: np.ndarray, Y_resid: np.ndarray) -> Stage2Result:
        result = self.fn(T_k, Y_resid)
        if not isinstance(result, Stage2Result):
            raise TypeError(
                f"CustomEstimator fn must return Stage2Result, got {type(result)}"
            )
        return result


class HSICEstimator(Stage2Estimator):
    """
    Stage-2 HSIC (Hilbert-Schmidt Independence Criterion) test.

    Kernel-based nonparametric independence test between Tₖ and Ỹ.
    Detects nonlinear and non-monotone dependence that OLS cannot capture.
    The statistic is unsigned (always ≥ 0), so use alongside OLSEstimator
    when direction of effect matters.

    Parameters
    ----------
    n_permutations : int, default=500
        Permutations for the p-value. 500 gives reliable results; use 1000+
        for publication-quality estimates.
    bandwidth : float or None, default=None
        RBF kernel bandwidth σ. None applies the median heuristic
        (σ = median of pairwise distances), which is a robust default.
    alpha : float, default=0.05
    random_state : int or None, default=None

    Notes
    -----
    beta and t_stat are both set to the raw HSIC statistic (scale depends on
    bandwidth and n). They are comparable across representations fitted with
    the same criterion instance but not across different datasets or bandwidths.
    """

    def __init__(
        self,
        n_permutations: int = 500,
        bandwidth: Optional[float] = None,
        alpha: float = 0.05,
        random_state: Optional[int] = None,
    ):
        self.n_permutations = n_permutations
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, T_k: np.ndarray, Y_resid: np.ndarray) -> Stage2Result:
        T_k = np.asarray(T_k, dtype=float).ravel()
        Y_resid = np.asarray(Y_resid, dtype=float).ravel()

        K = _rbf_kernel(T_k, self.bandwidth)
        L = _rbf_kernel(Y_resid, self.bandwidth)
        observed = _hsic_statistic(K, L)

        if self.n_permutations == 0:
            # Informational mode: statistic only, no p-value
            return Stage2Result(
                beta=observed,
                t_stat=observed,
                pvalue=float("nan"),
                significant=False,
                method="hsic",
                model_result=None,
            )

        rng = np.random.default_rng(self.random_state)
        null = np.empty(self.n_permutations)
        for i in range(self.n_permutations):
            perm = rng.permutation(len(Y_resid))
            null[i] = _hsic_statistic(K, L[np.ix_(perm, perm)])

        # +1 conservative correction for finite permutations
        pvalue = float((np.sum(null >= observed) + 1) / (self.n_permutations + 1))

        return Stage2Result(
            beta=observed,
            t_stat=observed,
            pvalue=pvalue,
            significant=pvalue < self.alpha,
            method="hsic",
            model_result=None,
        )
