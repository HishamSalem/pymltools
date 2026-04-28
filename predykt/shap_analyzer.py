"""
SHAP Interaction Analyzer
=========================
Three composable layers of SHAP attribution that correct for collinearity
and cross-group interaction aliasing.

THE PROBLEM:
    When xᵢ, xⱼ, Tₖ = f(xᵢ, xⱼ) are all in the model, raw SHAP values are
    aliased: apparent importance of each feature depends on tree structure,
    not true contribution. Cross-group attribution is contaminated.

THE SOLUTION: Three layers of progressively purer attribution.

THEORETICAL FOUNDATIONS:
    Lundberg & Lee (2017): SHAP efficiency axiom
        Σᵢ SHAP(xᵢ) = f(x) - E[f(x)]
        Group sums are valid by this axiom regardless of collinearity.

    Lundberg et al. (2018): SHAP interaction values
        shap_interaction_values[obs, i, i] = pure main effect of feature i
        shap_interaction_values[obs, i, j] = pairwise interaction i↔j
        Off-diagonal entries are symmetric: [i,j] == [j,i]
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union


class SHAPInteractionAnalyzer:
    """
    Three-layer SHAP attribution with collinearity and cross-group aliasing correction.

    Parameters
    ----------
    interaction_groups : dict
        Maps group name to list of feature names in that group.
        MUST cover ALL features in the model. No orphans, no overlaps.
        This is enforced in fit(). Required for Layer 2 cross-group stripping.

        Example:
            {
                "liquidity": ["current_ratio", "quick_ratio"],
                "leverage":  ["debt_equity", "debt_assets"],
                "temporal":  ["hour_bin", "month"],
            }

    layers : list of int, default=[1, 2, 3]
        Which layers to compute. Any subset of [1, 2, 3].
        Layer 1 data is always computed internally even if 1 is not in this
        list; it is required by Layer 2.

    Attributes
    ----------
    layer1_ : pd.DataFrame, shape (n_samples, n_groups)
        Group total SHAP: sum of shap_values within each group.
    layer2_ : pd.DataFrame, shape (n_samples, n_groups)
        Net group effects: Layer 1 minus cross-group interaction contributions.
    layer3_ : pd.DataFrame, shape (n_samples, n_features)
        Pure main effects: diagonal of shap_interaction_values (feature-level).
    feature_names_ : list of str
    group_feature_indices_ : dict, {group_name: [int indices]}
    """

    def __init__(
        self,
        interaction_groups: Dict[str, List[str]],
        layers: List[int] = None,
    ):
        self.interaction_groups = interaction_groups
        self.layers = layers if layers is not None else [1, 2, 3]

        self.layer1_ = None
        self.layer2_ = None
        self.layer3_ = None
        self.feature_names_ = None
        self.group_feature_indices_ = None

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_groups(self, feature_names: List[str]) -> None:
        """
        Enforce complete, non-overlapping group coverage.
        Layer 2 requires that every feature belongs to exactly one group.
        """
        feature_set = set(feature_names)
        all_assigned: List[str] = []
        for g, feats in self.interaction_groups.items():
            all_assigned.extend(feats)

        assigned_set = set(all_assigned)

        # Orphan features
        orphans = feature_set - assigned_set
        if orphans:
            raise ValueError(
                f"interaction_groups does not cover all features. "
                f"Unassigned features: {sorted(orphans)}. "
                "Layer 2 requires complete group membership. "
                "Add a catch-all group (e.g. 'other') for ungrouped features."
            )

        # Extra features in groups not in data
        extras = assigned_set - feature_set
        if extras:
            raise ValueError(
                f"interaction_groups references features not in shap_values: "
                f"{sorted(extras)}."
            )

        # Duplicate assignments
        if len(all_assigned) != len(assigned_set):
            from collections import Counter
            counts = Counter(all_assigned)
            duplicates = {f for f, c in counts.items() if c > 1}
            raise ValueError(
                f"Features appear in multiple groups: {sorted(duplicates)}. "
                "Each feature must belong to exactly one group."
            )

    def _build_group_indices(self, feature_names: List[str]) -> Dict[str, List[int]]:
        fn_list = list(feature_names)
        return {
            g: [fn_list.index(f) for f in feats]
            for g, feats in self.interaction_groups.items()
        }

    # =========================================================================
    # LAYER COMPUTATIONS
    # =========================================================================

    def _compute_layer1(self, shap_values: np.ndarray) -> pd.DataFrame:
        """
        Layer 1: Group Total.

        shap_group[obs, g] = sum(shap_values[obs, f] for f in F_g)

        Always valid. SHAP efficiency axiom guarantees this is preserved
        regardless of collinearity within the group.
        """
        data = {
            g: shap_values[:, indices].sum(axis=1)
            for g, indices in self.group_feature_indices_.items()
        }
        return pd.DataFrame(data)

    def _compute_layer2(
        self,
        shap_interaction_values: np.ndarray,
        layer1_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Layer 2: Net Group Effects.

        For each group g:
            cross_g = sum_{f in F_g, h NOT in F_g} shap_interaction_values[:, f, h]
            layer2_g = layer1_g - cross_g / 2

        The /2 corrects for off-diagonal symmetry: [f,h] == [h,f]. Each
        cross-group interaction is represented once in group g's sum from f's
        perspective; dividing by 2 strips the interaction's shared attribution.

        Comparing Layer 1 vs Layer 2 per group quantifies how much of the
        group's apparent importance is driven by cross-group interactions.
        """
        _, p, _ = shap_interaction_values.shape
        data = {}

        for g, f_indices in self.group_feature_indices_.items():
            out_indices = [i for i in range(p) if i not in f_indices]

            f_idx = np.array(f_indices)
            h_idx = np.array(out_indices)
            cross = shap_interaction_values[:, f_idx[:, None], h_idx[None, :]].sum(axis=(1, 2))
            cross = cross / 2.0

            data[g] = layer1_data[g].values - cross

        return pd.DataFrame(data)

    def _compute_layer3(
        self,
        shap_values: np.ndarray,
        shap_interaction_values: np.ndarray,
    ) -> pd.DataFrame:
        """
        Layer 3: Pure Main Effects (feature-level).

        shap_main[obs, f] = shap_interaction_values[obs, f, f]

        The diagonal entry is the pure main effect of feature f net of ALL
        interactions with every other feature in the model.

        Output shape is (n_samples, n_features), not group-level.

        Consistency check: sum of all diagonal + off-diagonal entries should
        equal sum of shap_values (SHAP efficiency). Warns if violated.
        """
        _, p, _ = shap_interaction_values.shape

        # Diagonal: pure main effects
        main_effects = np.stack(
            [shap_interaction_values[:, i, i] for i in range(p)],
            axis=1,
        )  # shape (n, p)

        # Off-diagonal sum: total sum minus diagonal, no double counting
        diagonal_sum = main_effects.sum(axis=1)
        off_diag = shap_interaction_values.sum(axis=(1, 2)) - diagonal_sum

        reconstructed = diagonal_sum + off_diag
        shap_sum = shap_values.sum(axis=1)
        max_diff = float(np.max(np.abs(reconstructed - shap_sum)))

        if max_diff > 1e-4:
            warnings.warn(
                f"Layer 3 consistency check failed: "
                f"shap_interaction_values are inconsistent with shap_values. "
                f"max|reconstructed - shap_sum| = {max_diff:.2e}. "
                "This may indicate a binary classifier where the wrong class "
                "index was passed. For TreeExplainer on binary classifiers, "
                "pass shap_values[1] and shap_interaction_values[1].",
                UserWarning,
                stacklevel=3,
            )

        return pd.DataFrame(main_effects, columns=self.feature_names_)

    # =========================================================================
    # FIT
    # =========================================================================

    def fit(
        self,
        model=None,
        X: Optional[pd.DataFrame] = None,
        shap_values: Optional[np.ndarray] = None,
        shap_interaction_values: Optional[np.ndarray] = None,
    ) -> "SHAPInteractionAnalyzer":
        """
        Compute SHAP values and all requested layers.

        Two modes:

        Mode A: internal computation (model + X):
            model : fitted sklearn-compatible tree model
            X     : pd.DataFrame, shape (n, p)

        Mode B: external precomputed values:
            shap_values             : np.ndarray, shape (n, p)
            shap_interaction_values : np.ndarray, shape (n, p, p)

        For binary classifiers, TreeExplainer returns lists (one per class).
        Always pass index [1] when providing precomputed values in Mode B.
        In Mode A, this is handled automatically.

        Parameters
        ----------
        model : fitted tree model, optional (Mode A)
        X : pd.DataFrame, optional (Mode A)
        shap_values : np.ndarray, optional (Mode B)
        shap_interaction_values : np.ndarray, optional (Mode B, required for layers 2/3)

        Returns
        -------
        self
        """
        needs_interaction = any(l in self.layers for l in [2, 3])

        if shap_values is not None:
            # Mode B
            sv = np.asarray(shap_values, dtype=float)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            siv = None
            if needs_interaction:
                if shap_interaction_values is None:
                    raise ValueError(
                        "shap_interaction_values is required for layers 2 or 3. "
                        "Provide precomputed values or use Mode A (model + X)."
                    )
                siv = np.asarray(shap_interaction_values, dtype=float)
                if siv.ndim == 4:
                    siv = siv[:, :, :, 1]

            n, p = sv.shape
            if X is not None:
                feature_names = list(X.columns)
            else:
                feature_names = [f"f{i}" for i in range(p)]
        else:
            # Mode A
            if model is None or X is None:
                raise ValueError(
                    "Provide (model, X) for Mode A or shap_values for Mode B."
                )
            import shap as shap_lib
            feature_names = list(X.columns)
            explainer = shap_lib.TreeExplainer(model)

            sv_raw = explainer.shap_values(X)
            if isinstance(sv_raw, list):
                sv_raw = sv_raw[1]
            sv = np.asarray(sv_raw, dtype=float)
            if sv.ndim == 3:
                sv = sv[:, :, 1]

            siv = None
            if needs_interaction:
                siv_raw = explainer.shap_interaction_values(X)
                if isinstance(siv_raw, list):
                    siv_raw = siv_raw[1]
                siv = np.asarray(siv_raw, dtype=float)
                if siv.ndim == 4:
                    siv = siv[:, :, :, 1]

        self.feature_names_ = feature_names
        self._validate_groups(feature_names)
        self.group_feature_indices_ = self._build_group_indices(feature_names)

        # Layer 1: always stored (cheap, needed internally for Layer 2)
        l1_data = self._compute_layer1(sv)
        self.layer1_ = l1_data

        if 2 in self.layers:
            self.layer2_ = self._compute_layer2(siv, l1_data)

        if 3 in self.layers:
            self.layer3_ = self._compute_layer3(sv, siv)

        return self

    # =========================================================================
    # OUTPUT
    # =========================================================================

    def layer_1_group_total(self) -> pd.DataFrame:
        """
        Layer 1: group total SHAP (n_samples, n_groups).
        Always valid regardless of collinearity.
        """
        if self.layer1_ is None:
            raise RuntimeError("Layer 1 not computed. Include 1 in layers=.")
        return self.layer1_.copy()

    def layer_2_net_group_effects(self) -> pd.DataFrame:
        """
        Layer 2: net group effects (n_samples, n_groups).
        Group total minus cross-group interaction contributions.
        """
        if self.layer2_ is None:
            raise RuntimeError("Layer 2 not computed. Include 2 in layers=.")
        return self.layer2_.copy()

    def layer_3_pure_main_effects(self) -> pd.DataFrame:
        """
        Layer 3: pure main effects (n_samples, n_features).
        Diagonal of shap_interaction_values, net of all pairwise interactions.
        """
        if self.layer3_ is None:
            raise RuntimeError("Layer 3 not computed. Include 3 in layers=.")
        return self.layer3_.copy()

    def summary(
        self,
        layer: int,
        aggregate: str = "mean_abs",
    ) -> pd.DataFrame:
        """
        Global importance summary for a given layer.

        Parameters
        ----------
        layer : int
            1, 2, or 3.
        aggregate : str, default="mean_abs"
            "mean_abs": mean(|SHAP|) across observations. Standard.
            "mean":     signed mean. Use to see directionality.
            "sum_abs":  total absolute contribution across all observations.

        Returns
        -------
        pd.DataFrame with columns [group/feature, importance],
        sorted descending by importance.

        Diagnostic use:
            Layer 1 importance - Layer 2 importance per group
                = cross-group interaction contribution.
                Large delta → group's apparent importance is driven by
                interactions with other groups, not the concept itself.

            Layer 2 importance - sum(Layer 3 importance within group)
                = within-group interaction contribution.
                Large delta → collinearity within group is aliasing individual
                feature importances even after cross-group correction.
        """
        layer_map = {1: self.layer1_, 2: self.layer2_, 3: self.layer3_}
        df = layer_map.get(layer)
        if df is None:
            raise RuntimeError(
                f"Layer {layer} not computed. Include {layer} in layers= and call fit()."
            )

        if aggregate == "mean_abs":
            importance = df.abs().mean()
        elif aggregate == "mean":
            importance = df.mean()
        elif aggregate == "sum_abs":
            importance = df.abs().sum()
        else:
            raise ValueError(
                f"aggregate must be 'mean_abs', 'mean', or 'sum_abs'. Got '{aggregate}'."
            )

        col_label = "feature" if layer == 3 else "group"
        result = importance.rename("importance").reset_index()
        result.columns = [col_label, "importance"]
        return result.sort_values("importance", ascending=False).reset_index(drop=True)

    def compare_layers(
        self, aggregate: str = "mean_abs"
    ) -> tuple:
        """
        Side-by-side importance across all computed layers.

        Layers 1 and 2 are group-level; Layer 3 is feature-level. Because
        their indices differ they cannot be safely concatenated into a single
        DataFrame; this method returns two separate DataFrames.

        Parameters
        ----------
        aggregate : str, default="mean_abs"
            Same options as summary(): "mean_abs", "mean", "sum_abs".

        Returns
        -------
        group_comparison : pd.DataFrame
            Columns layer_1 and/or layer_2, indexed by group name.
            layer_1 - layer_2 per group = cross-group interaction contribution.
        feature_main_effects : pd.DataFrame
            Column layer_3, indexed by feature name. Empty DataFrame if
            Layer 3 was not computed.
        """
        group_frames = {}
        if self.layer1_ is not None:
            group_frames["layer_1"] = (
                self.summary(1, aggregate).set_index("group")["importance"]
            )
        if self.layer2_ is not None:
            group_frames["layer_2"] = (
                self.summary(2, aggregate).set_index("group")["importance"]
            )

        feature_frame = pd.DataFrame()
        if self.layer3_ is not None:
            feature_frame = (
                self.summary(3, aggregate)
                .set_index("feature")[["importance"]]
                .rename(columns={"importance": "layer_3"})
            )

        if not group_frames and feature_frame.empty:
            raise RuntimeError("No layers computed. Call fit() first.")

        group_comparison = (
            pd.concat(group_frames, axis=1).rename_axis(None, axis=0)
            if group_frames
            else pd.DataFrame()
        )
        return group_comparison, feature_frame
