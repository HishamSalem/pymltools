"""
Feature Binning Analyzer using Information Value (IV).

Uses optimal 1D and 2D binning (via optbinning) to quantify individual and
joint predictive power of feature pairs. Intended as a screening tool in the
early stage of a feature interaction pipeline — it identifies which pairs
carry joint information beyond their marginal contributions. Not a substitute
for the formal residual-based interaction test in ResidualRepresentationTester.
"""

import numpy as np
import pandas as pd
from optbinning import OptimalBinning, OptimalBinning2D


class FeatureBinningAnalyzer:
    """
    Screen feature pairs for interaction signal via Information Value uplift.

    Fits optimal binning on each feature individually (1D IV) and jointly
    (2D IV), then computes uplift = IV_2D - (IV_1 + IV_2). A positive uplift
    indicates the pair carries joint predictive information not captured by
    either feature alone.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix. Column names must be strings.
    y : pd.Series or np.ndarray
        Binary target (1 = event, 0 = non-event). Must be 1-dimensional.

    Attributes
    ----------
    results : pd.DataFrame or None
        Set after calling analyze_feature_combinations(). Contains one row
        per feature pair with IV_1, IV_2, IV_2D, Uplift, and Binning_Table.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        self.X = X
        self.y = y
        self.results: pd.DataFrame | None = None

    def get_feature_iv(self, feature_name: str) -> float:
        """
        Compute the 1D Information Value for a single feature.

        Parameters
        ----------
        feature_name : str
            Column name in self.X.

        Returns
        -------
        float
            Maximum IV from the fitted binning table.

        Raises
        ------
        KeyError
            If feature_name is not a column in self.X.
        """
        if feature_name not in self.X.columns:
            raise KeyError(
                f"Feature '{feature_name}' not found in X. "
                f"Available columns: {list(self.X.columns)}"
            )
        optb = OptimalBinning(name=feature_name, solver="cp")
        optb.fit(self.X[feature_name], self.y)
        binning_table = optb.binning_table.build()
        return binning_table['IV'].max()

    def get_2d_feature_iv(
        self, feature1: str, feature2: str
    ) -> tuple[float, pd.DataFrame]:
        """
        Compute the joint 2D Information Value for a feature pair.

        Parameters
        ----------
        feature1 : str
            First feature column name in self.X.
        feature2 : str
            Second feature column name in self.X.

        Returns
        -------
        total_iv : float
            Maximum IV from the 2D binning table.
        binning_table : pd.DataFrame
            Full 2D binning table from optbinning.

        Raises
        ------
        KeyError
            If feature1 or feature2 is not a column in self.X.
        """
        for fname in (feature1, feature2):
            if fname not in self.X.columns:
                raise KeyError(
                    f"Feature '{fname}' not found in X. "
                    f"Available columns: {list(self.X.columns)}"
                )
        optb = OptimalBinning2D(name_x=feature1, name_y=feature2, solver="cp")
        optb.fit(self.X[feature1].values, self.X[feature2].values, self.y)
        binning_table = optb.binning_table.build()
        total_iv = binning_table['IV'].max()
        return total_iv, binning_table

    def analyze_feature_combinations(
        self, feature_pairs: list[tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Compute IV and uplift for a list of feature pairs.

        For each pair (feat1, feat2), computes IV_1, IV_2 (marginal IVs),
        IV_2D (joint IV), and Uplift = IV_2D - (IV_1 + IV_2). Stores the
        full results table in self.results for downstream access.

        Parameters
        ----------
        feature_pairs : list of (str, str)
            Feature pairs to evaluate. Both elements must be column names in self.X.

        Returns
        -------
        pd.DataFrame
            One row per pair. Columns: Feature1, Feature2, IV_1, IV_2,
            IV_2D, Uplift, Binning_Table.
        """
        results = []

        for feat1, feat2 in feature_pairs:
            iv1 = self.get_feature_iv(feat1)
            iv2 = self.get_feature_iv(feat2)
            iv_2d, binning_table = self.get_2d_feature_iv(feat1, feat2)

            sum_iv = iv1 + iv2
            uplift = iv_2d - sum_iv

            results.append({
                'Feature1': feat1,
                'Feature2': feat2,
                'IV_1': iv1,
                'IV_2': iv2,
                'IV_2D': iv_2d,
                'Uplift': uplift,
                'Binning_Table': binning_table
            })

        self.results = pd.DataFrame(results)
        return self.results

    def get_top_combinations(self) -> pd.DataFrame:
        """
        Return feature pairs sorted by uplift descending.

        Returns
        -------
        pd.DataFrame
            Columns: Feature1, Feature2, IV_1, IV_2, IV_2D, Uplift.
            Sorted by Uplift descending (highest interaction signal first).

        Raises
        ------
        ValueError
            If analyze_feature_combinations has not been called.
        """
        if self.results is None:
            raise ValueError("Run analyze_feature_combinations first")

        sorted_results = self.results.sort_values('Uplift', ascending=False)
        return sorted_results[['Feature1', 'Feature2', 'IV_1', 'IV_2', 'IV_2D', 'Uplift']]

    def get_binning_details(self, feature1: str, feature2: str) -> pd.DataFrame:
        """
        Retrieve the 2D binning table for a specific feature pair.

        Parameters
        ----------
        feature1 : str
            First feature of the pair.
        feature2 : str
            Second feature of the pair.

        Returns
        -------
        pd.DataFrame
            The 2D binning table stored during analyze_feature_combinations.

        Raises
        ------
        ValueError
            If analyze_feature_combinations has not been called.
        """
        if self.results is None:
            raise ValueError("Run analyze_feature_combinations first")
        mask = (self.results['Feature1'] == feature1) & (self.results['Feature2'] == feature2)
        return self.results[mask]['Binning_Table'].iloc[0]
