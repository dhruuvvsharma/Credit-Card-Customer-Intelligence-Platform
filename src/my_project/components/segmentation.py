import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# from sklearn.decomposition import PCA
import pickle
import json

from src.my_project.exceptions import CustomException
from src.my_project.logger import logging


@dataclass
class SegmentationConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    cluster_labels_path: str = os.path.join("artifacts", "cluster_labels.csv")
    kmeans_model_path: str = os.path.join("artifacts", "kmeans_model.pkl")
    scaler_path: str = os.path.join("artifacts", "scaler.pkl")
    # pca_model_path: str = os.path.join("artifacts", "pca.pkl")
    silhouette_score_path: str = os.path.join("artifacts", "metrics", "silhouette_score.json")

    # Locked feature set from EDA 
    behavior_features: tuple = (
        "Total_Revolving_Bal",
        "Avg_Utilization_Ratio",
        "Total_Trans_Ct",
        "Total_Trans_Amt",
        "Contacts_Count_12_mon"
    )

    # skew check log transform 
    skewed_features: tuple = ("Total_Trans_Amt",)

    id_column: str = "CLIENTNUM"
    n_clusters: int = 4          # placeholder: confirm via elbow/silhouette sweep
    random_state: int = 42


class CustomerSegmentation:
    def __init__(self, config: SegmentationConfig = SegmentationConfig()):
        self.config = config

    def load_data(self):
        try:
            df = pd.read_csv(self.config.raw_data_path)
            logging.info(f"Raw data loaded for segmentation: shape: {df.shape}")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select locked behavior features and log transform the skewed ones"""
        features = df[list(self.config.behavior_features)].copy()

        for col in self.config.skewed_features:
            # log1p handles zero values safely
            features[col] = np.log1p(features[col])

        return features

    def scale_features(self, features: pd.DataFrame):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        return scaled, scaler

    def find_optimal_k(self, scaled_features, k_range=range(2, 8)):
        """Sweep k, return silhouette scores for each, inspect before locking n_clusters"""
        scores = {}
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
            labels = km.fit_predict(scaled_features)
            score = silhouette_score(scaled_features, labels)
            scores[k] = score
        return scores

    def fit_kmeans(self, scaled_features):
        kmeans = KMeans(n_clusters=self.config.n_clusters,random_state=self.config.random_state,n_init=10,)
        cluster_ids = kmeans.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, cluster_ids)
        return kmeans, cluster_ids, score

    def save_artifacts(self, df, cluster_ids, kmeans, scaler, silhouette):
        os.makedirs(os.path.dirname(self.config.cluster_labels_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.silhouette_score_path), exist_ok=True)

        # cluster_labels.csv: CLIENTNUM + Cluster_ID, for the merge into train/test later
        labels_df = df[[self.config.id_column]].copy()   # original dataframe se sirf CLIENTNUM column le lo
        labels_df["Cluster_ID"] = cluster_ids
        labels_df.to_csv(self.config.cluster_labels_path, index=False)

        with open(self.config.kmeans_model_path, "wb") as f:
            pickle.dump(kmeans, f)

        with open(self.config.scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        with open(self.config.silhouette_score_path, "w") as f:
            json.dump({"silhouette_score": float(silhouette)}, f, indent=2)

    def initiate_segmentation(self):
        logging.info("Segmentation started")
        try:
            df = self.load_data()
            features = self.prepare_features(df)
            scaled_features, scaler = self.scale_features(features)

            kmeans, cluster_ids, silhouette = self.fit_kmeans(scaled_features)
            self.save_artifacts(df, cluster_ids, kmeans, scaler, silhouette)
            logging.info(f"Segmentation completed, silhouette score: {silhouette:.4f}")

            return self.config.cluster_labels_path, silhouette
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    segmentation = CustomerSegmentation()
    labels_path, score = segmentation.initiate_segmentation()
    print(f"Cluster labels saved to: {labels_path}")
    print(f"Silhouette score: {score:.4f}")