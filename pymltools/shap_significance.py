import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import shap

class SHAPInteractionSignificance:
    """
    A class for testing statistical significance of SHAP interaction values using parallel processing
    and optimized calculations. Supports binary classification models.

    Parameters
    ----------
    model : object
        Trained tree-based model that implements predict
    data : pandas.DataFrame
        Input data for analysis
    random_state : int, optional
        Random seed for reproducibility
    """
    def __init__(self, model, data, random_state=None):
        self.model = model
        self.data = data
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not hasattr(model, 'predict'):
            raise TypeError("model must implement predict method")
            
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            
        # Initialize faster SHAP explainer
        self.explainer = shap.TreeExplainer(
            model,
            approximate=True,
        )
        self.data_columns = data.columns.tolist()
        self.shap_interaction_values = self.explainer.shap_interaction_values(data)

    def _calculate_permuted_interaction(self, feature1, feature2, data):
        """Helper function for parallel processing of permuted interactions"""
        permuted_data = data.copy()
        permuted_data[feature1] = np.random.permutation(permuted_data[feature1])
        permuted_data[feature2] = np.random.permutation(permuted_data[feature2])
        
        permuted_explainer = shap.TreeExplainer(
            self.model, 
            approximate=True,
        )
        permuted_interactions_matrix = permuted_explainer.shap_interaction_values(permuted_data)
        
        index1 = self.data_columns.index(feature1)
        index2 = self.data_columns.index(feature2)
        return permuted_interactions_matrix[:, index1, index2].mean()

    def calculate_original_interaction_value(self, feature1, feature2):
        """Calculate the original SHAP interaction value for a feature pair"""
        index1 = self.data_columns.index(feature1)
        index2 = self.data_columns.index(feature2)
        interaction_values = self.shap_interaction_values[:, index1, index2]
        return interaction_values.mean()

    def get_top_n_interactions(self, n=10):
        """Get the top N feature pairs based on absolute interaction values"""
        interactions = []
        for i, feat1 in enumerate(self.data_columns):
            for feat2 in self.data_columns[i+1:]:
                value = abs(self.calculate_original_interaction_value(feat1, feat2))
                interactions.append((feat1, feat2, value))
        
        # Sort by absolute interaction value and get top N
        top_pairs = sorted(interactions, key=lambda x: abs(x[2]), reverse=True)[:n]
        return [(pair[0], pair[1]) for pair in top_pairs]

    def test_significance(self, feature_pairs=None, n_permutations=1000, alpha=0.05, 
                         correction_method='fdr_bh', n_jobs=-1):
        """
        Test significance of non-zero feature interactions using parallel permutation testing
        
        Parameters
        ----------
        feature_pairs : list of tuples, optional
            List of (feature1, feature2) tuples to test. If None, tests all pairs
        n_permutations : int
            Number of permutations for the test
        alpha : float
            Significance level
        correction_method : str
            Multiple testing correction method
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        """
        n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        results = []
        
        # If no specific pairs provided, create pairs from all features
        if feature_pairs is None:
            feature_pairs = []
            for i, feat1 in enumerate(self.data_columns):
                for feat2 in self.data_columns[i+1:]:
                    feature_pairs.append((feat1, feat2))
        
        for feature1, feature2 in tqdm(feature_pairs):
            original_value = self.calculate_original_interaction_value(feature1, feature2)
            
            # Skip if interaction value is zero
            if np.isclose(original_value, 0, atol=1e-10):
                continue
                
            # Parallel permutation processing
            permuted_values = Parallel(n_jobs=n_jobs)(
                delayed(self._calculate_permuted_interaction)(
                    feature1, feature2, self.data
                ) for _ in range(n_permutations)
            )
            
            p_value = np.mean(np.abs(permuted_values) >= np.abs(original_value))
            
            results.append({
                'Feature1': feature1,
                'Feature2': feature2,
                'Original Interaction Value': original_value,
                'p-value': p_value,
            })

        results_df = pd.DataFrame(results)
        
        # Only perform multiple testing correction if we have results
        if len(results_df) > 0:
            reject, corrected_p_values, _, _ = multipletests(
                results_df['p-value'], alpha=alpha, method=correction_method
            )
            results_df['Adjusted p-value'] = corrected_p_values
            results_df['Significant'] = reject
        
        return results_df

    def plot_null_distribution(self, feature1, feature2, n_permutations=1000, n_jobs=-1):
        """Plot the null distribution of interaction values with the observed value"""
        original_value = self.calculate_original_interaction_value(feature1, feature2)
        
        # Generate null distribution using parallel processing
        n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        null_distribution = Parallel(n_jobs=n_jobs)(
            delayed(self._calculate_permuted_interaction)(
                feature1, feature2, self.data
            ) for _ in range(n_permutations)
        )

        plt.figure(figsize=(10, 6))
        sns.histplot(null_distribution, kde=True)
        plt.axvline(original_value, color='r', linestyle='--', label='Observed Value')
        plt.title(f'Null Distribution of SHAP Interaction Values\n{feature1} and {feature2}')
        plt.xlabel('SHAP Interaction Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    def plot_top_interactions(self, results_df, top_n=10, color='#1f77b4'):
        """Plot top feature interactions by absolute interaction value"""
        # Create copy and add absolute value column
        plot_df = results_df.copy()
        plot_df['Absolute Interaction Value'] = plot_df['Original Interaction Value'].abs()
        
        # Get top N based on absolute values
        top_interactions = plot_df.nlargest(top_n, 'Absolute Interaction Value').copy()
        top_interactions['Feature Pair'] = top_interactions['Feature1'] + ' + ' + top_interactions['Feature2']
        
        plt.figure(figsize=(12, 8))
        sns.barplot(
            x='Absolute Interaction Value', 
            y='Feature Pair', 
            data=top_interactions, 
            color=color
        )
        plt.title(f'Top {top_n} Feature Interactions by Absolute SHAP Interaction Value')
        plt.xlabel('Absolute SHAP Interaction Value')
        plt.ylabel('Feature Pair')
        plt.tight_layout()
        plt.show()


