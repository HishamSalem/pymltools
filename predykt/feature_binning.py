from optbinning import OptimalBinning, OptimalBinning2D
import pandas as pd

class FeatureBinningAnalyzer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.results = None

    def get_feature_iv(self, feature_name):
        optb = OptimalBinning(name=feature_name, solver="cp")
        optb.fit(self.X[feature_name], self.y)
        binning_table = optb.binning_table.build()  # Build the table
        return binning_table['IV'].max()

    def get_2d_feature_iv(self, feature1, feature2):
        optb = OptimalBinning2D(name_x=feature1, name_y=feature2, solver="cp")
        optb.fit(self.X[feature1].values, self.X[feature2].values, self.y)
        binning_table = optb.binning_table.build()
        total_iv = binning_table['IV'].max()
        return total_iv, binning_table



    def analyze_feature_combinations(self, feature_pairs):
        results = []

        for feat1, feat2 in feature_pairs:
            # Individual IVs
            iv1 = self.get_feature_iv(feat1)
            iv2 = self.get_feature_iv(feat2)

            # 2D IV
            iv_2d, binning_table = self.get_2d_feature_iv(feat1, feat2)

            # Calculate uplift
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

    def get_top_combinations(self):
        if self.results is None:
            raise ValueError("Run analyze_feature_combinations first")

        # Sort all results by uplift in descending order
        sorted_results = self.results.sort_values('Uplift', ascending=False)

        # Return relevant columns for better readability
        return sorted_results[['Feature1', 'Feature2', 'IV_1', 'IV_2', 'IV_2D', 'Uplift']]


    def get_binning_details(self, feature1, feature2):
        if self.results is None:
            raise ValueError("Run analyze_feature_combinations first")
        mask = (self.results['Feature1'] == feature1) & (self.results['Feature2'] == feature2)
        return self.results[mask]['Binning_Table'].iloc[0]