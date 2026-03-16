"""
SHAP Interaction Significance via Bootstrap Stability Testing
=============================================================

Tests whether SHAP interaction values are statistically significant
by measuring their stability across multiple model fits with varied
random seeds, then validating across algorithm families.

Replaces the permutation-on-fixed-model approach (which tests the wrong
null hypothesis) with a refit-across-seeds approach that tests whether
the interaction is a stable property of the data-generating process.

Core idea:
    - A real interaction should be stable across random seeds (p-value)
    - A real interaction should have predictive power (per-interaction AUC)
    - A real interaction should appear across multiple algorithms (vote)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from joblib import Parallel, delayed
from tqdm import tqdm
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import multiprocessing


# =============================================================================
# RESULT CONTAINERS
# =============================================================================

@dataclass
class InteractionResult:
    """Result for a single feature pair from a single algorithm."""
    feature_i: str
    feature_j: str
    algorithm: str
    mean_interaction: float
    std_interaction: float
    p_value: float
    per_interaction_auc: float
    mean_auc: float
    std_auc: float
    n_seeds: int
    significant: bool
    interaction_distribution: np.ndarray = field(repr=False)
    auc_distribution: np.ndarray = field(repr=False)


@dataclass
class VoteResult:
    """Cross-algorithm vote result for a single feature pair."""
    feature_i: str
    feature_j: str
    n_votes: int
    n_algorithms: int
    vote_ratio: float
    algorithm_results: Dict[str, InteractionResult]
    unanimous: bool
    mean_auc_across_algorithms: float

    def __repr__(self):
        status = "UNANIMOUS" if self.unanimous else f"{self.n_votes}/{self.n_algorithms}"
        return (
            f"VoteResult({self.feature_i} x {self.feature_j}: "
            f"{status}, mean_auc={self.mean_auc_across_algorithms:.4f})"
        )


# =============================================================================
# CORE: SINGLE ALGORITHM INTERACTION TESTER
# =============================================================================

class InteractionTester:
    """
    Test SHAP interaction significance for a single algorithm by
    refitting across multiple random seeds and measuring stability.

    Parameters
    ----------
    model_class : class
        Unfitted model class (e.g., XGBClassifier).
    base_params : dict
        Frozen hyperparameters. Must NOT include the random seed param.
    seed_param : str
        Name of the random seed parameter for this model class.
    n_seeds : int
        Number of random seeds to fit. Default 200.
    alpha : float
        Significance threshold for p-values.
    use_gpu : bool
        Whether to use GPU-accelerated SHAP explainer.
    n_jobs : int
        Number of parallel jobs for seed fitting. -1 for all cores.
    """

    def __init__(
        self,
        model_class,
        base_params: dict,
        seed_param: str = "random_state",
        n_seeds: int = 200,
        alpha: float = 0.05,
        use_gpu: bool = False,
        n_jobs: int = 1,
    ):
        self.model_class = model_class
        self.base_params = base_params
        self.seed_param = seed_param
        self.n_seeds = n_seeds
        self.alpha = alpha
        self.use_gpu = use_gpu
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

        # Validate seed_param not in base_params
        if seed_param in base_params:
            raise ValueError(
                f"'{seed_param}' should not be in base_params. "
                f"It will be set automatically per seed."
            )

    def _fit_single_seed(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        seed: int,
        pair_indices: List[Tuple[int, int]],
    ) -> Dict:
        """
        Fit model with given seed, compute SHAP interactions,
        extract metrics for requested pairs only.

        Returns dict with per-pair mean interaction and AUC.
        """
        params = {**self.base_params, self.seed_param: seed}
        model = self.model_class(**params)
        model.fit(X, y)

        # Compute SHAP interaction values
        if self.use_gpu:
            try:
                explainer = shap.explainers.GPUTree(
                    model, X,
                    feature_perturbation="tree_path_dependent",
                )
                interactions = explainer(X, interactions=True)
            except Exception:
                warnings.warn(
                    f"GPU explainer failed for seed {seed}, falling back to CPU."
                )
                explainer = shap.TreeExplainer(model)
                interactions = explainer.shap_interaction_values(X)
        else:
            explainer = shap.TreeExplainer(model)
            interactions = explainer.shap_interaction_values(X)

        # Handle binary classifiers returning list per class
        if isinstance(interactions, list):
            interactions = interactions[1]

        # Extract metrics for each requested pair
        pair_results = {}
        for idx_i, idx_j in pair_indices:
            # Mean interaction value across samples for this pair
            interaction_vals = interactions[:, idx_i, idx_j]
            mean_val = float(np.mean(interaction_vals))

            # Per-interaction AUC: can this pair's SHAP interaction
            # values alone discriminate the target?
            try:
                auc = roc_auc_score(y, interaction_vals)
                auc = max(auc, 1 - auc)  # direction-invariant
            except ValueError:
                auc = 0.5

            pair_results[(idx_i, idx_j)] = {
                "mean_interaction": mean_val,
                "auc": auc,
            }

        return pair_results

    def _compute_p_value(self, distribution: np.ndarray) -> float:
        """
        Empirical p-value: test if bootstrap distribution is centered at zero.
        Two-sided sign-based test.
        """
        observed_mean = np.mean(distribution)
        if observed_mean > 0:
            p = np.mean(distribution <= 0)
        else:
            p = np.mean(distribution >= 0)
        return float(min(p * 2, 1.0))

    def test_pairs(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_pairs: List[Tuple[str, str]],
        seeds: Optional[np.ndarray] = None,
    ) -> List[InteractionResult]:
        """
        Test multiple feature pairs across n_seeds model fits.

        Computes all pairs per seed in a single pass to avoid
        redundant model fitting and SHAP computation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : array-like
            Binary target.
        feature_pairs : list of (str, str)
            Feature pairs to test.
        seeds : array-like, optional
            Specific seeds to use. If None, uses range(n_seeds).

        Returns
        -------
        List of InteractionResult, one per pair.
        """
        if seeds is None:
            seeds = np.arange(self.n_seeds)

        y = np.asarray(y).ravel()
        columns = X.columns.tolist()

        # Map feature names to indices once
        pair_indices = []
        for feat_i, feat_j in feature_pairs:
            idx_i = columns.index(feat_i)
            idx_j = columns.index(feat_j)
            pair_indices.append((idx_i, idx_j))

        # Run all seeds — each seed computes all pairs in one pass
        if self.n_jobs > 1:
            all_seed_results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_single_seed)(X, y, int(seed), pair_indices)
                for seed in tqdm(seeds, desc=f"{self.model_class.__name__}")
            )
        else:
            all_seed_results = [
                self._fit_single_seed(X, y, int(seed), pair_indices)
                for seed in tqdm(seeds, desc=f"{self.model_class.__name__}")
            ]

        # Assemble distributions per pair
        results = []
        for p_idx, (feat_i, feat_j) in enumerate(feature_pairs):
            idx_i, idx_j = pair_indices[p_idx]

            interaction_dist = np.array([
                sr[(idx_i, idx_j)]["mean_interaction"] for sr in all_seed_results
            ])
            auc_dist = np.array([
                sr[(idx_i, idx_j)]["auc"] for sr in all_seed_results
            ])

            p_value = self._compute_p_value(interaction_dist)

            results.append(InteractionResult(
                feature_i=feat_i,
                feature_j=feat_j,
                algorithm=self.model_class.__name__,
                mean_interaction=float(np.mean(interaction_dist)),
                std_interaction=float(np.std(interaction_dist)),
                p_value=p_value,
                per_interaction_auc=float(np.mean(auc_dist)),
                mean_auc=float(np.mean(auc_dist)),
                std_auc=float(np.std(auc_dist)),
                n_seeds=len(seeds),
                significant=p_value < self.alpha,
                interaction_distribution=interaction_dist,
                auc_distribution=auc_dist,
            ))

        return results

    def get_top_n_interactions(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n: int = 10,
        seed: int = 42,
    ) -> List[Tuple[str, str]]:
        """
        Quick screening: single-fit SHAP interactions to identify
        candidate pairs for full bootstrap testing.

        Use this to pre-filter before running the expensive test_pairs.
        """
        params = {**self.base_params, self.seed_param: seed}
        model = self.model_class(**params)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        interactions = explainer.shap_interaction_values(X)
        if isinstance(interactions, list):
            interactions = interactions[1]

        columns = X.columns.tolist()
        pair_scores = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                mean_abs = float(np.abs(interactions[:, i, j]).mean())
                pair_scores.append((columns[i], columns[j], mean_abs))

        pair_scores.sort(key=lambda x: x[2], reverse=True)
        return [(p[0], p[1]) for p in pair_scores[:n]]

    def results_to_dataframe(
        self,
        results: List[InteractionResult],
        correction_method: Optional[str] = "fdr_bh",
    ) -> pd.DataFrame:
        """
        Convert results to DataFrame with optional multiple testing correction.
        """
        df = pd.DataFrame([
            {
                "Feature_i": r.feature_i,
                "Feature_j": r.feature_j,
                "Algorithm": r.algorithm,
                "Mean_Interaction": r.mean_interaction,
                "Std_Interaction": r.std_interaction,
                "P_Value": r.p_value,
                "Per_Interaction_AUC": r.per_interaction_auc,
                "Std_AUC": r.std_auc,
                "Significant": r.significant,
                "N_Seeds": r.n_seeds,
            }
            for r in results
        ])

        if correction_method and len(df) > 1:
            reject, adj_p, _, _ = multipletests(
                df["P_Value"], alpha=self.alpha, method=correction_method
            )
            df["Adjusted_P_Value"] = adj_p
            df["Significant_Adjusted"] = reject

        return df

    # =========================================================================
    # PLOTTING
    # =========================================================================

    def plot_interaction_distribution(
        self,
        result: InteractionResult,
        figsize: Tuple[int, int] = (10, 6),
    ):
        """Plot bootstrap distribution of interaction values with zero reference."""
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

        # Left: interaction value distribution
        ax = axes[0]
        sns.histplot(result.interaction_distribution, kde=True, ax=ax)
        ax.axvline(0, color="r", linestyle="--", label="H0: zero", alpha=0.7)
        ax.axvline(
            result.mean_interaction, color="g", linestyle="-",
            label=f"Mean: {result.mean_interaction:.6f}", alpha=0.7,
        )
        ax.set_title(
            f"Interaction Distribution: {result.feature_i} x {result.feature_j}\n"
            f"{result.algorithm} | p={result.p_value:.4f} | n_seeds={result.n_seeds}"
        )
        ax.set_xlabel("Mean SHAP Interaction Value")
        ax.set_ylabel("Frequency")
        ax.legend()

        # Right: per-interaction AUC distribution
        ax = axes[1]
        sns.histplot(result.auc_distribution, kde=True, ax=ax)
        ax.axvline(0.5, color="r", linestyle="--", label="H0: no discrimination", alpha=0.7)
        ax.axvline(
            result.mean_auc, color="g", linestyle="-",
            label=f"Mean AUC: {result.mean_auc:.4f}", alpha=0.7,
        )
        ax.set_title(
            f"Per-Interaction AUC: {result.feature_i} x {result.feature_j}\n"
            f"{result.algorithm} | Std: {result.std_auc:.4f}"
        )
        ax.set_xlabel("AUC (interaction term only)")
        ax.set_ylabel("Frequency")
        ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_convergence(
        self,
        result: InteractionResult,
        figsize: Tuple[int, int] = (12, 5),
    ):
        """
        Plot running mean and std of interaction value across seeds.
        Useful for determining if n_seeds is sufficient.
        """
        dist = result.interaction_distribution
        running_mean = np.cumsum(dist) / np.arange(1, len(dist) + 1)
        running_std = np.array([
            np.std(dist[:i+1]) for i in range(len(dist))
        ])

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].plot(running_mean)
        axes[0].axhline(result.mean_interaction, color="r", linestyle="--", alpha=0.5)
        axes[0].set_title(f"Convergence: {result.feature_i} x {result.feature_j}")
        axes[0].set_xlabel("Number of Seeds")
        axes[0].set_ylabel("Running Mean Interaction")

        axes[1].plot(running_std)
        axes[1].set_title("Running Std")
        axes[1].set_xlabel("Number of Seeds")
        axes[1].set_ylabel("Std of Interaction")

        plt.tight_layout()
        plt.show()

    def plot_top_interactions(
        self,
        results_df: pd.DataFrame,
        top_n: int = 10,
        color: str = "#1f77b4",
        figsize: Tuple[int, int] = (12, 8),
    ):
        """Plot top feature interactions by per-interaction AUC."""
        plot_df = results_df.nlargest(top_n, "Per_Interaction_AUC").copy()
        plot_df["Feature Pair"] = plot_df["Feature_i"] + " x " + plot_df["Feature_j"]

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            x="Per_Interaction_AUC", y="Feature Pair",
            data=plot_df, color=color, ax=ax,
        )
        ax.axvline(0.5, color="r", linestyle="--", alpha=0.5, label="No discrimination")
        ax.set_title(f"Top {top_n} Interactions by Per-Interaction AUC")
        ax.set_xlabel("Mean Per-Interaction AUC (across seeds)")
        ax.legend()
        plt.tight_layout()
        plt.show()


# =============================================================================
# CROSS-ALGORITHM VOTER
# =============================================================================

class InteractionVoter:
    """
    Run interaction testing across multiple algorithms and vote.

    Parameters
    ----------
    algorithm_configs : dict
        Mapping of name -> dict with:
            "model_class": unfitted model class
            "params": frozen hyperparameters (no seed param)
            "seed_param": name of random seed parameter
    n_seeds : int
        Number of seeds per algorithm.
    alpha : float
        Significance threshold.
    use_gpu : bool
        GPU acceleration for SHAP.
    n_jobs : int
        Parallel jobs per algorithm.
    """

    def __init__(
        self,
        algorithm_configs: Dict[str, Dict[str, Any]],
        n_seeds: int = 200,
        alpha: float = 0.05,
        use_gpu: bool = False,
        n_jobs: int = 1,
    ):
        self.algorithm_configs = algorithm_configs
        self.n_seeds = n_seeds
        self.alpha = alpha

        self.testers = {}
        for name, config in algorithm_configs.items():
            self.testers[name] = InteractionTester(
                model_class=config["model_class"],
                base_params=config["params"],
                seed_param=config.get("seed_param", "random_state"),
                n_seeds=n_seeds,
                alpha=alpha,
                use_gpu=use_gpu,
                n_jobs=n_jobs,
            )

    def vote(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_pairs: List[Tuple[str, str]],
        seeds: Optional[np.ndarray] = None,
    ) -> List[VoteResult]:
        """
        Test each feature pair across all algorithms and tally votes.

        A pair receives a vote from an algorithm if it is significant
        (p < alpha) after the bootstrap stability test.
        """
        all_results = {}
        for algo_name, tester in self.testers.items():
            print(f"\n{'='*60}")
            print(f"  {algo_name} ({tester.model_class.__name__})")
            print(f"{'='*60}")
            results = tester.test_pairs(X, y, feature_pairs, seeds)
            for r in results:
                all_results[(algo_name, r.feature_i, r.feature_j)] = r

        # Tally votes per pair
        vote_results = []
        n_algorithms = len(self.testers)

        for feat_i, feat_j in feature_pairs:
            algo_results = {}
            votes = 0
            aucs = []

            for algo_name in self.testers:
                r = all_results[(algo_name, feat_i, feat_j)]
                algo_results[algo_name] = r
                if r.significant:
                    votes += 1
                aucs.append(r.per_interaction_auc)

            vote_results.append(VoteResult(
                feature_i=feat_i,
                feature_j=feat_j,
                n_votes=votes,
                n_algorithms=n_algorithms,
                vote_ratio=votes / n_algorithms,
                algorithm_results=algo_results,
                unanimous=(votes == n_algorithms),
                mean_auc_across_algorithms=float(np.mean(aucs)),
            ))

        return sorted(
            vote_results,
            key=lambda v: (-v.n_votes, -v.mean_auc_across_algorithms),
        )

    def summary(self, vote_results: List[VoteResult]) -> pd.DataFrame:
        """Summary DataFrame from vote results."""
        rows = []
        for vr in vote_results:
            row = {
                "Feature_i": vr.feature_i,
                "Feature_j": vr.feature_j,
                "Votes": vr.n_votes,
                "Total_Algorithms": vr.n_algorithms,
                "Vote_Ratio": vr.vote_ratio,
                "Unanimous": vr.unanimous,
                "Mean_AUC": vr.mean_auc_across_algorithms,
            }
            for algo_name, r in vr.algorithm_results.items():
                row[f"{algo_name}_p"] = r.p_value
                row[f"{algo_name}_auc"] = r.per_interaction_auc
                row[f"{algo_name}_significant"] = r.significant
            rows.append(row)

        return pd.DataFrame(rows)

    def plot_vote_heatmap(
        self,
        vote_results: List[VoteResult],
        figsize: Tuple[int, int] = (14, 8),
    ):
        """Heatmap: algorithms (columns) x feature pairs (rows), colored by AUC."""
        algo_names = list(self.testers.keys())
        pair_labels = [f"{vr.feature_i} x {vr.feature_j}" for vr in vote_results]

        auc_matrix = np.zeros((len(vote_results), len(algo_names)))
        sig_matrix = np.zeros((len(vote_results), len(algo_names)), dtype=bool)

        for i, vr in enumerate(vote_results):
            for j, algo in enumerate(algo_names):
                r = vr.algorithm_results[algo]
                auc_matrix[i, j] = r.per_interaction_auc
                sig_matrix[i, j] = r.significant

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            auc_matrix,
            xticklabels=algo_names,
            yticklabels=pair_labels,
            annot=True, fmt=".3f",
            cmap="RdYlGn", center=0.5,
            vmin=0.45, vmax=0.7,
            ax=ax,
        )

        # Mark significant cells
        for i in range(sig_matrix.shape[0]):
            for j in range(sig_matrix.shape[1]):
                if sig_matrix[i, j]:
                    ax.text(
                        j + 0.5, i + 0.85, "*",
                        ha="center", va="center",
                        fontsize=14, fontweight="bold", color="black",
                    )

        ax.set_title("Per-Interaction AUC by Algorithm (* = significant)")
        plt.tight_layout()
        plt.show()


# =============================================================================
# USAGE
# =============================================================================

# if __name__ == "__main__":
#     from xgboost import XGBClassifier
#     from lightgbm import LGBMClassifier
#     from catboost import CatBoostClassifier
#     from sklearn.ensemble import RandomForestClassifier

#     # ---- Frozen hyperparameters per algorithm ----
#     configs = {
#         "rf": {
#             "model_class": RandomForestClassifier,
#             "params": {"n_estimators": 200, "max_depth": 5, "n_jobs": -1},
#             "seed_param": "random_state",
#         },
#         "xgb": {
#             "model_class": XGBClassifier,
#             "params": {
#                 "n_estimators": 200, "max_depth": 5,
#                 "use_label_encoder": False, "eval_metric": "logloss",
#                 "verbosity": 0,
#             },
#             "seed_param": "random_state",
#         },
#         "lgbm": {
#             "model_class": LGBMClassifier,
#             "params": {
#                 "n_estimators": 200, "max_depth": 5, "verbose": -1,
#             },
#             "seed_param": "random_state",
#         },
#         "catboost": {
#             "model_class": CatBoostClassifier,
#             "params": {
#                 "iterations": 200, "depth": 5, "verbose": 0,
#             },
#             "seed_param": "random_seed",
#         },
#     }

    # ---- Single algorithm quick test ----
    # tester = InteractionTester(
    #     model_class=XGBClassifier,
    #     base_params=configs["xgb"]["params"],
    #     n_seeds=200,
    #     n_jobs=4,
    # )
    # top_pairs = tester.get_top_n_interactions(X, y, n=10)
    # results = tester.test_pairs(X, y, top_pairs)
    # df = tester.results_to_dataframe(results)
    # tester.plot_interaction_distribution(results[0])
    # tester.plot_convergence(results[0])

    # ---- Full cross-algorithm vote ----
    # voter = InteractionVoter(configs, n_seeds=200, alpha=0.05, n_jobs=4)
    # vote_results = voter.vote(X, y, top_pairs)
    # print(voter.summary(vote_results))
    # voter.plot_vote_heatmap(vote_results)
