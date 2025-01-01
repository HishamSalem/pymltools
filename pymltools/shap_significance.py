import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import shap

class SHAPInteractionSignificance:
    """
    A class for testing statistical significance of SHAP interaction values. Only supports a binary classification model.

    Parameters
    ----------
    model : object
        Trained tree-based model that implements predict
    data : pandas.DataFrame
    random_state : int, optional
        Random seed for reproducibility
    """
    def __init__(self, model, data, random_state=None):
        self.model = model
        self.data = data
        self.explainer = shap.TreeExplainer(model)
        self.shap_interaction_values = self.explainer.shap_interaction_values(data)
        self.data_columns = data.columns.tolist()
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not hasattr(model, 'predict'):
            raise TypeError("model must implement predict method")


    def calculate_original_interaction_value(self, feature1, feature2):
        index1 = self.data_columns.index(feature1)
        index2 = self.data_columns.index(feature2)
        interaction_values = self.shap_interaction_values[:, index1, index2]
        interaction_value = interaction_values.mean()
        return interaction_value

    def test_significance(self, interaction_df, n_permutations=1000, alpha=0.05, correction_method='fdr_bh'):
        results = []
        for _, row in tqdm(interaction_df.iterrows(), total=interaction_df.shape[0], desc="Testing significance"):
            feature1 = row['Feature1']
            feature2 = row['Feature2']
            original_interaction_value = self.calculate_original_interaction_value(feature1, feature2)
            
            # Get SHAP interaction values for the two features
            index1 = self.data_columns.index(feature1)
            index2 = self.data_columns.index(feature2)
            shap_values = self.shap_interaction_values[:, index1, index2]

            # Perform permutation test
            res = permutation_test(
                data=(shap_values, np.zeros_like(shap_values)),
                statistic=lambda x, y: np.mean(x - y),
                permutation_type='independent',
                vectorized=False,
                n_resamples=n_permutations,
                alternative='two-sided',
                random_state=self.random_state
            )

            p_value = res.pvalue

            result = {
                'Feature1': feature1,
                'Feature2': feature2,
                'Original Interaction Value': original_interaction_value,
                'p-value': p_value,
            }
            results.append(result)

        results_df = pd.DataFrame(results)
        
        # Apply multiple testing correction
        reject, corrected_p_values, _, _ = multipletests(
            results_df['p-value'], alpha=alpha, method=correction_method
        )
        results_df['Adjusted p-value'] = corrected_p_values
        results_df['Significant'] = reject

        return results_df

    def plot_null_distribution(self, feature1, feature2, n_permutations=1000):
        original_value = self.calculate_original_interaction_value(feature1, feature2)
        index1 = self.data_columns.index(feature1)
        index2 = self.data_columns.index(feature2)
        shap_values = self.shap_interaction_values[:, index1, index2]

        # Perform permutation test to get null distribution
        res = permutation_test(
            data=(shap_values, np.zeros_like(shap_values)),
            statistic=lambda x, y: np.mean(x - y),
            permutation_type='independent',
            vectorized=False,
            n_resamples=n_permutations,
            alternative='two-sided',
            random_state=self.random_state
        )

        permuted_values = res.null_distribution

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.histplot(permuted_values, kde=True)
        plt.axvline(original_value, color='r', linestyle='--', label='Observed Value')
        plt.title(f'Null Distribution of SHAP Interaction Values for {feature1} and {feature2}')
        plt.xlabel('SHAP Interaction Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    def plot_top_interactions(self, results_df, top_n=10, color='#1f77b4'):  # Default matplotlib blue
        top_interactions = results_df.nlargest(top_n, 'Original Interaction Value').copy()
        top_interactions['Feature Pair'] = top_interactions['Feature1'] + ' + ' + top_interactions['Feature2']
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Original Interaction Value', y='Feature Pair', 
                    data=top_interactions, color=color)
        plt.title(f'Top {top_n} Feature Interactions by SHAP Interaction Value')
        plt.xlabel('SHAP Interaction Value')
        plt.ylabel('Feature Pair')
        plt.tight_layout()
        plt.show()


