import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler


class FeatureSelector:
    def __init__(self, correlation_threshold=0.85, importance_percentile=20):
        """
        Initialize feature selector with given thresholds.

        Parameters:
        correlation_threshold: Maximum allowed correlation between features
        importance_percentile: Percentile of features to keep based on importance
        """
        self.correlation_threshold = correlation_threshold
        self.importance_percentile = importance_percentile
        self.selected_features = None
        self.feature_importances = None

    def remove_correlated_features(self, X, method="spearman"):
        """
        Remove highly correlated features using hierarchical clustering.

        Parameters:
        X: pandas DataFrame of features
        method: Correlation method ('spearman' or 'pearson')

        Returns:
        DataFrame with correlated features removed
        """
        # Calculate correlation matrix
        corr = X.corr(method=method)

        # Convert correlation matrix to distance matrix
        distance_matrix = 1 - np.abs(corr)

        # Perform hierarchical clustering
        Z = hierarchy.linkage(
            hierarchy.distance.squareform(distance_matrix), method="average"
        )

        # Form flat clusters with distance threshold
        clusters = hierarchy.fcluster(
            Z, t=1 - self.correlation_threshold, criterion="distance"
        )

        # For each cluster, keep feature with highest variance
        selected_features = []
        for cluster_id in np.unique(clusters):
            cluster_features = np.where(clusters == cluster_id)[0]
            if len(cluster_features) == 1:
                selected_features.append(X.columns[cluster_features[0]])
            else:
                variances = X.iloc[:, cluster_features].var()
                selected_features.append(variances.idxmax())

        return X[selected_features]

    def calculate_feature_importance(self, X, y):
        """
        Calculate feature importance using multiple methods.

        Parameters:
        X: Feature matrix
        y: Target labels (cluster assignments)

        Returns:
        DataFrame with feature importance scores
        """
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns)

        # Mutual Information
        mi_importance = pd.Series(
            mutual_info_classif(X, y, random_state=42), index=X.columns
        )

        # ANOVA F-value
        f_importance = pd.Series(f_classif(X, y)[0], index=X.columns)

        # Combine all methods
        importance_df = pd.DataFrame(
            {
                "random_forest": rf_importance,
                "mutual_information": mi_importance,
                "f_score": f_importance,
            }
        )

        # Normalize each column
        importance_df = importance_df.apply(lambda x: x / x.max())

        # Calculate mean importance
        importance_df["mean_importance"] = importance_df.mean(axis=1)

        return importance_df.sort_values(
            "mean_importance", ascending=False
        )  # pyright: ignore

    def fit(self, X, y):
        """
        Fit the feature selector to the data.

        Parameters:
        X: Feature matrix
        y: Target labels (cluster assignments)
        """
        # Remove correlated features
        X_uncorrelated = self.remove_correlated_features(X)

        # Calculate feature importance
        self.feature_importances = self.calculate_feature_importance(X_uncorrelated, y)

        # Select top features based on importance
        n_features = int(
            len(self.feature_importances) * self.importance_percentile / 100
        )
        self.selected_features = self.feature_importances.head(
            n_features
        ).index.tolist()

    def transform(self, X):
        """
        Transform data using selected features.

        Parameters:
        X: Feature matrix

        Returns:
        Transformed feature matrix
        """
        return X[self.selected_features]

    def fit_transform(self, X, y):
        """
        Fit and transform the data.

        Parameters:
        X: Feature matrix
        y: Target labels (cluster assignments)

        Returns:
        Transformed feature matrix
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        self.fit(X_scaled, y)
        return self.transform(X)

    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance scores.

        Parameters:
        top_n: Number of top features to display
        """
        plt.figure(figsize=(12, 6))

        # Plot top N features
        top_features = self.feature_importances.head(top_n)  # pyright: ignore
        sns.barplot(
            x="mean_importance", y=top_features.index, data=top_features.reset_index()
        )

        plt.title(f"Top {top_n} Most Important Features")
        plt.xlabel("Mean Importance Score")
        plt.tight_layout()

    def plot_correlation_matrix(self, X, subset=None):
        """
        Plot correlation matrix for selected features.

        Parameters:
        X: Original feature matrix
        subset: Number of top features to display
        """
        if subset is None:
            features_to_plot = self.selected_features
        else:
            features_to_plot = self.selected_features[:subset]  # pyright: ignore

        corr = X[features_to_plot].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Matrix of Selected Features")
        plt.tight_layout()
