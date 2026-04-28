"""
Residual Representation Tester
==============================
Tests whether a user-engineered representation of a confirmed interaction pair
carries statistically significant signal BEYOND what the base model already
learned, using a residual-based approach as an honest baseline.

Theory
------
Frisch-Waugh-Lovell (1933): In Y ~ β·T + γ·X, the coefficient β equals that
from regressing partialled-out Ỹ on partialled-out T̃.

Extended nonparametrically (Chernozhukov et al. 2018, Double/Debiased ML):
  Stage 1: Fit base model on full X via K-fold cross-fitting.
           Compute OOF residuals: Ỹ = y − p̂.
  Stage 2: Regress Ỹ on Tₖ and test H₀: β₁ = 0.

T residualization degeneracy (this is correct, not a limitation):
  Tₖ = f(xᵢ, xⱼ) is a deterministic function of X.
  Therefore E[Tₖ | X] = Tₖ and T̃ₖ = 0.
  Only outcome residualization is needed.

What a significant β₁ means:
  The base model already learned an interaction between xᵢ and xⱼ.
  Ỹ = what it failed to explain.
  β₁ ≠ 0 → the functional form Tₖ captures structure the model missed.

Critical assumption:
  Base model must be well-specified and not underfit. If the model is weak,
  Ỹ is large for unrelated reasons and β₁ would be spuriously inflated.

Multiple testing correction:
  Only meaningful when K ≥ 2 representations are tested simultaneously.
  When K = 1 the raw p-value is the answer and correction is a no-op.

Refutation:
  Call refute() after fit() to add placebo permutation p-values and bootstrap
  stability scores. robust = rejected AND passes both refutation checks.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Optional, Tuple, Union

from .criteria import Stage2Estimator, OLSEstimator, Stage2Result

logger = logging.getLogger(__name__)


class ResidualRepresentationTester:
    """
    Residual-based representation validator.

    Parameters
    ----------
    model : sklearn-compatible classifier with predict_proba
        Stage-1 model. Cloned per fold. Required unless Y_resid is passed to fit().
    criterion : Stage2Estimator or list of Stage2Estimator
        Stage-2 criterion. Pass OLSEstimator(), HSICEstimator(), or a list of
        both to run multiple criteria. Defaults to OLSEstimator(HC3).
        ``stage2`` is accepted as a backward-compatible alias.
    n_folds : int, default=5
        K-fold cross-fitting folds.
    correction_method : str, default="fdr_bh"
        Multiple testing correction applied when K ≥ 2 representations are
        tested. "fdr_bh" (Benjamini-Hochberg) or "bonferroni". No-op when K=1.
    correction_scope : str, default="per_pair"
        Scope for multiple testing correction. "per_pair" applies correction
        within each (pair, criterion) group independently.
    alpha : float, default=0.05
    random_state : int or None, default=42

    Attributes
    ----------
    Y_resid_ : np.ndarray, shape (n,)
        OOF residuals Ỹ = y − p̂.
    results_ : list of dict
        One dict per (pair, criterion, representation).
    feature_pairs_ : list of tuples
    """

    def __init__(
        self,
        model=None,
        stage2: Optional[Stage2Estimator] = None,
        n_folds: int = 5,
        correction_method: str = "fdr_bh",
        correction_scope: str = "per_pair",
        alpha: float = 0.05,
        random_state: Optional[int] = 42,
        criterion=None,
    ):
        self.model = model
        resolved = criterion if criterion is not None else stage2
        if resolved is None:
            resolved = OLSEstimator(alpha=alpha)
        self.criteria: List[Stage2Estimator] = (
            resolved if isinstance(resolved, list) else [resolved]
        )
        self.stage2 = self.criteria[0]  # backward-compat alias
        self.n_folds = n_folds
        self.correction_method = correction_method
        self.correction_scope = correction_scope
        self.alpha = alpha
        self.random_state = random_state

        self.Y_resid_: Optional[np.ndarray] = None
        self.results_: Optional[List[dict]] = None
        self.feature_pairs_: Optional[List[Tuple]] = None
        self._T_k_map_: Dict[Tuple, np.ndarray] = {}
        self._criteria_by_method_: Dict[str, Stage2Estimator] = {}

    # =========================================================================
    # STAGE 1: OOF RESIDUALS
    # =========================================================================

    def _compute_residuals(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        p_hat = np.zeros_like(y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(X):
            mdl = clone(self.model)
            X_tr = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
            X_va = X.iloc[val_idx]   if isinstance(X, pd.DataFrame) else X[val_idx]
            mdl.fit(X_tr, y[train_idx])
            p_hat[val_idx] = mdl.predict_proba(X_va)[:, 1]

        return y - p_hat

    # =========================================================================
    # FIT
    # =========================================================================

    def fit(
        self,
        feature_pairs: List[Tuple[str, str]],
        X: pd.DataFrame,
        y: np.ndarray,
        representations: Union[pd.DataFrame, Dict[Tuple, pd.DataFrame]],
        Y_resid: Optional[np.ndarray] = None,
    ) -> "ResidualRepresentationTester":
        """
        Compute OOF residuals (stage 1) and test all representations (stage 2).

        Parameters
        ----------
        feature_pairs : list of (str, str)
        X : pd.DataFrame
        y : np.ndarray
        representations : pd.DataFrame or dict of pd.DataFrame
            Single pair → pd.DataFrame (each column is one representation Tₖ).
            Multiple pairs → dict keyed by (feat_i, feat_j).
        Y_resid : np.ndarray, optional
            Precomputed OOF residuals. Skips stage-1 computation.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=float).ravel()
        feature_pairs = [tuple(p) for p in feature_pairs]
        self.feature_pairs_ = feature_pairs

        # --- Stage 1 ---
        if Y_resid is not None:
            self.Y_resid_ = np.asarray(Y_resid, dtype=float).ravel()
        else:
            if self.model is None:
                raise ValueError(
                    "Provide model= (Mode A: OOF residuals computed here) "
                    "or Y_resid= (Mode B: precomputed residuals)."
                )
            logger.info("Stage 1: computing OOF residuals (%d-fold)...", self.n_folds)
            self.Y_resid_ = self._compute_residuals(X, y)

        # --- Normalise representations ---
        if isinstance(representations, pd.DataFrame):
            if len(feature_pairs) != 1:
                raise ValueError(
                    "Pass representations as a dict when testing multiple pairs."
                )
            rep_dict = {feature_pairs[0]: representations}
        else:
            rep_dict = {tuple(k): v for k, v in representations.items()}

        # --- Stage 2: run all criteria for all representations ---
        raw_rows: List[dict] = []
        self._T_k_map_ = {}
        self._criteria_by_method_ = {}

        for pair in feature_pairs:
            if pair not in rep_dict:
                raise ValueError(
                    f"No representations provided for pair {pair}. "
                    f"Available keys: {list(rep_dict.keys())}"
                )
            rep_df = rep_dict[pair]

            for col in rep_df.columns:
                T_k = rep_df[col].values.astype(float)
                self._T_k_map_[(pair, col)] = T_k

                for crit in self.criteria:
                    res: Stage2Result = crit.fit(T_k, self.Y_resid_)
                    self._criteria_by_method_[res.method] = crit

                    raw_rows.append({
                        "pair":             pair,
                        "criterion":        res.method,
                        "representation":   col,
                        "beta":             res.beta,
                        "statistic":        res.t_stat,
                        "pvalue":           res.pvalue,
                        "pvalue_bh":        None,
                        "rejected":         False,
                        "robust":           None,
                        "winner":           False,
                        "empirical_pvalue": None,
                        "stability_score":  None,
                        "stage2_result":    res,
                    })

        # --- Multiple testing correction ---
        def _apply_bh(rows_to_correct: List[dict]) -> None:
            valid_mask = [
                r["pvalue"] is not None
                and not (isinstance(r["pvalue"], float) and np.isnan(r["pvalue"]))
                for r in rows_to_correct
            ]
            valid_rows   = [r for r, ok in zip(rows_to_correct, valid_mask) if ok]
            invalid_rows = [r for r, ok in zip(rows_to_correct, valid_mask) if not ok]
            valid_pvals  = [r["pvalue"] for r in valid_rows]

            for r in invalid_rows:
                r["pvalue_bh"] = float("nan")
                r["rejected"]  = False

            if len(valid_pvals) == 1:
                valid_rows[0]["pvalue_bh"] = float(valid_pvals[0])
                valid_rows[0]["rejected"]  = float(valid_pvals[0]) < self.alpha
            elif len(valid_pvals) > 1:
                reject, padj, _, _ = multipletests(
                    valid_pvals, alpha=self.alpha, method=self.correction_method
                )
                for row, rej, pc in zip(valid_rows, reject, padj):
                    row["pvalue_bh"] = float(pc)
                    row["rejected"]  = bool(rej)

        if self.correction_scope == "global":
            _apply_bh(raw_rows)
        else:
            # per_pair: correct within each (pair, criterion) group independently
            groups: Dict[tuple, List[dict]] = {}
            for row in raw_rows:
                groups.setdefault((row["pair"], row["criterion"]), []).append(row)
            for rows in groups.values():
                _apply_bh(rows)

        # Winner: rejected + largest |statistic| per (pair, criterion)
        winner_groups: Dict[tuple, List[dict]] = {}
        for row in raw_rows:
            winner_groups.setdefault((row["pair"], row["criterion"]), []).append(row)
        for rows in winner_groups.values():
            sig = [r for r in rows if r["rejected"]]
            if sig:
                max(sig, key=lambda r: abs(r["statistic"]))["winner"] = True

        self.results_ = raw_rows
        return self

    # =========================================================================
    # REFUTATION
    # =========================================================================

    def refute(
        self,
        n_permutations: int = 100,
        n_bootstrap: int = 50,
        stability_threshold: float = 0.8,
    ) -> "ResidualRepresentationTester":
        """
        Refutation checks for each (pair, criterion, representation).

        1. Placebo permutation test: permutes Y_resid and re-runs stage-2.
           empirical_pvalue = P(|t_perm| ≥ |t_obs|) under H₀.
           A large empirical_pvalue means the signal could be noise.

        2. Bootstrap stability: re-runs stage-2 on 80% subsamples.
           stability_score = fraction of bootstrap runs where p < alpha.

        3. robust = rejected AND empirical_pvalue < alpha AND
                    stability_score ≥ stability_threshold.

        Parameters
        ----------
        n_permutations : int, default=100
        n_bootstrap : int, default=50
        stability_threshold : float, default=0.8

        Returns
        -------
        self
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() first.")

        rng = np.random.default_rng(self.random_state)
        n = len(self.Y_resid_)
        subsample_size = max(int(0.8 * n), 10)

        for row in self.results_:
            T_k   = self._T_k_map_.get((row["pair"], row["representation"]))
            crit  = self._criteria_by_method_.get(row["criterion"])
            pval  = row["pvalue"]

            has_valid_pval = (
                pval is not None
                and not (isinstance(pval, float) and np.isnan(pval))
            )

            if T_k is None or crit is None or not has_valid_pval:
                row["empirical_pvalue"] = float("nan")
                row["stability_score"]  = float("nan")
                row["robust"]           = None  # no p-value, refutation not applicable
                continue

            obs_stat = abs(row["statistic"])

            # 1. Permutation placebo
            null_stats = np.empty(n_permutations)
            for i in range(n_permutations):
                perm = rng.permutation(n)
                res_p = crit.fit(T_k, self.Y_resid_[perm])
                t = res_p.t_stat
                null_stats[i] = abs(t) if t is not None and not np.isnan(t) else 0.0

            emp_pval = float(
                (np.sum(null_stats >= obs_stat) + 1) / (n_permutations + 1)
            )
            row["empirical_pvalue"] = emp_pval

            # 2. Bootstrap stability
            sig_count = 0
            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=subsample_size, replace=False)
                res_b = crit.fit(T_k[idx], self.Y_resid_[idx])
                bp = res_b.pvalue
                if bp is not None and not np.isnan(bp) and bp < self.alpha:
                    sig_count += 1
            stab = sig_count / n_bootstrap
            row["stability_score"] = stab

            row["robust"] = bool(
                row["rejected"]
                and emp_pval < self.alpha
                and stab >= stability_threshold
            )

        return self

    # =========================================================================
    # OUTPUT
    # =========================================================================

    _DISPLAY_COLS = [
        "pair", "criterion", "representation", "beta",
        "statistic", "pvalue", "pvalue_bh", "rejected", "robust",
        "winner", "empirical_pvalue", "stability_score",
    ]

    def results_to_dataframe(
        self, pair: Optional[Tuple] = None
    ) -> pd.DataFrame:
        """
        Return results as a DataFrame, sorted by |statistic| descending.

        Columns
        -------
        pair, criterion, representation, beta, statistic, pvalue, pvalue_bh,
        rejected, robust, winner, empirical_pvalue, stability_score

        empirical_pvalue and stability_score are None until refute() is called.

        Parameters
        ----------
        pair : tuple, optional
            Filter to a single (feat_i, feat_j) pair.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() first.")

        rows = self.results_
        if pair is not None:
            rows = [r for r in rows if r["pair"] == tuple(pair)]

        df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "stage2_result"}
            for r in rows
        ])
        for col in self._DISPLAY_COLS:
            if col not in df.columns:
                df[col] = None

        df["_abs_stat"] = df["statistic"].abs()
        df = (
            df.sort_values(
                ["pair", "criterion", "rejected", "_abs_stat"],
                ascending=[True, True, False, False],
            )
            .drop(columns=["_abs_stat"])
            .reset_index(drop=True)
        )
        return df

    def winning_representations(self) -> Dict:
        """
        Return the winning representation per pair (primary criterion only).

        The primary criterion is the first element of the criteria list.
        A winner is the representation that (a) survived BH correction and
        (b) has the largest |statistic| for its (pair, criterion).

        Returns
        -------
        dict keyed by (feat_i, feat_j). Each value:
            representation   : str or None
            beta             : float or None
            statistic        : float or None
            pvalue_bh        : float or None
            rejected         : bool
            robust           : bool or None
            winner           : bool
            stage2_result    : Stage2Result or None

        Notes
        -----
        WINNER'S CURSE applies when K ≥ 2 representations were tested.
        The winning statistic is upward-biased. Use for ranking only.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() first.")

        primary_method = next(
            (r["criterion"] for r in self.results_ if r["pair"] == self.feature_pairs_[0]),
            None,
        )

        out = {}
        for pair in self.feature_pairs_:
            winners = [
                r for r in self.results_
                if r["pair"] == pair
                and r["criterion"] == primary_method
                and r["winner"]
            ]
            if winners:
                w = winners[0]
                out[pair] = {
                    "representation":  w["representation"],
                    "beta":            w["beta"],
                    "statistic":       w["statistic"],
                    "pvalue_bh":       w["pvalue_bh"],
                    "rejected":        w["rejected"],
                    "robust":          w["robust"],
                    "winner":          True,
                    "stage2_result":   w["stage2_result"],
                }
            else:
                out[pair] = {
                    "representation":  None,
                    "beta":            None,
                    "statistic":       None,
                    "pvalue_bh":       None,
                    "rejected":        False,
                    "robust":          None,
                    "winner":          False,
                    "stage2_result":   None,
                }
        return out
