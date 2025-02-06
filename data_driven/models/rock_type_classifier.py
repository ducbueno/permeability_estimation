# from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class RockTypeClassifier:
    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()

    def _plot_elbow_method(self, K, distortions, optimal_k):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(K, distortions, "bx-")
        plt.xlabel("k")
        plt.ylabel("Distortion")
        plt.title("Elbow Method")
        plt.vlines(
            x=optimal_k,
            ymin=min(distortions),
            ymax=max(distortions),
            colors="r",
            linestyles="dashed",
            label=f"Elbow at k={optimal_k}",
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _calculate_stability(self, X_scaled, k, n_iterations=10):
        cluster_assignments = []
        for _ in range(n_iterations):
            kmeans = KMeans(n_clusters=k, random_state=None)
            labels = kmeans.fit_predict(X_scaled)
            cluster_assignments.append(labels)

        cluster_sizes = np.array(
            [np.bincount(labels, minlength=k) for labels in cluster_assignments]
        )
        return 1 - np.mean(
            np.std(cluster_sizes, axis=0) / np.mean(cluster_sizes, axis=0)
        )

    def find_optimal_k(self, X, max_clusters=25, plot_result=False, verbose=False):
        X_scaled = self.scaler.fit_transform(X)

        distortions = []
        K = range(1, max_clusters + 1)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(X_scaled)
            distortions.append(kmeans.inertia_)

        kn = KneeLocator(K, distortions, curve="convex", direction="decreasing")
        optimal_k = kn.elbow

        if plot_result:
            self._plot_elbow_method(K, distortions, optimal_k)

        if verbose:
            stability_score = self._calculate_stability(X_scaled, optimal_k)
            print(f"Elbow Method suggests {optimal_k} clusters")
            print(f"Stability Score: {stability_score:.3f}")

        return optimal_k

    def fit_predict(self, X, n_clusters):
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        return self.kmeans.fit_predict(X_scaled)

    def predict(self, X):
        if self.kmeans is None:
            raise ValueError("Classifier must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
