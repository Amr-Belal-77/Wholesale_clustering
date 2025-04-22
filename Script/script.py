"""
script.py

This script performs clustering analysis (KMeans and DBSCAN) on the Wholesale customers dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import mode

class WholesaleClustering:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.cleaned_data = None
        self.standardized_data = None
        self.true_labels = None
        self.kmeans_labels = None
        self.dbscan_labels = None
        self.pca_data = None

    def remove_outliers_iqr(self):
        numeric_data = self.data.drop(columns=["Channel", "Region"])
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        self.cleaned_data = numeric_data[~((numeric_data < (Q1 - 1.5 * IQR)) | 
                                           (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)]
        self.true_labels = self.data.loc[self.cleaned_data.index, "Channel"]

    def standardize_data(self):
        scaler = StandardScaler()
        self.standardized_data = scaler.fit_transform(self.cleaned_data)

    def elbow_method(self, max_k=10):
        wcss = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.standardized_data)
            wcss.append(kmeans.inertia_)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_k + 1), wcss, 'bo-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS')
        plt.title('Elbow Method For Optimal k')
        plt.grid(True)
        plt.show()

    def run_kmeans(self, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans_labels = kmeans.fit_predict(self.standardized_data)

    def run_dbscan(self, eps=1.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.dbscan_labels = dbscan.fit_predict(self.standardized_data)

    def match_labels(self, pred_labels):
        labels = np.zeros_like(pred_labels)
        for i in np.unique(pred_labels):
            mask = (pred_labels == i)
            labels[mask] = mode(self.true_labels[mask])[0]
        return labels

    def evaluate_kmeans(self):
        matched = self.match_labels(self.kmeans_labels)
        acc = accuracy_score(self.true_labels, matched)
        cm = confusion_matrix(self.true_labels, matched)
        report = classification_report(self.true_labels, matched)

        print(f"🔍 Accuracy: {acc:.2f}")
        print("Confusion Matrix:\n", cm)
        print("\nClassification Report:\n", report)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def dbscan_summary(self):
        n_clusters = len(set(self.dbscan_labels)) - (1 if -1 in self.dbscan_labels else 0)
        n_noise = list(self.dbscan_labels).count(-1)
        print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points.")
        print("Sample DBSCAN labels:", self.dbscan_labels[:10])

    def visualize_clusters(self):
        pca = PCA(n_components=2)
        self.pca_data = pca.fit_transform(self.standardized_data)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=self.pca_data[:, 0], y=self.pca_data[:, 1],
                        hue=self.kmeans_labels, palette='Set2')
        plt.title("KMeans Clusters (PCA)")

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=self.pca_data[:, 0], y=self.pca_data[:, 1],
                        hue=self.dbscan_labels, palette='tab10')
        plt.title("DBSCAN Clusters (PCA)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    filepath = "X:/programming/DEPI_DS/ML/UnSupervised/Assignment_1/wholesale_clustering_project/Data/Wholesale customers data.csv"
    model = WholesaleClustering(filepath)

    model.remove_outliers_iqr()
    model.standardize_data()
    model.elbow_method()
    model.run_kmeans(n_clusters=3)
    model.evaluate_kmeans()
    model.run_dbscan(eps=1.5, min_samples=5)
    model.dbscan_summary()
    model.visualize_clusters()
