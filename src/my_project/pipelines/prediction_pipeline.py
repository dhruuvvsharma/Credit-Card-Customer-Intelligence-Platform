import os
import sys
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.my_project.exceptions import CustomException
from src.my_project.logger import logging


@dataclass
class PredictionPipelineConfig:
    kmeans_model_path: str = os.path.join("artifacts", "kmeans_model.pkl")
    segmentation_scaler_path: str = os.path.join("artifacts", "scaler.pkl")
    classification_preprocessor_path: str = os.path.join("artifacts", "classification_preprocessor.pkl")
    classification_model_path: str = os.path.join("artifacts", "classification_model.pkl")


    behavior_features: tuple = (
        "Total_Revolving_Bal", "Avg_Utilization_Ratio", "Total_Trans_Ct",
        "Total_Trans_Amt", "Contacts_Count_12_mon",
    )
    skewed_features: tuple = ("Total_Trans_Amt",)

    numeric_columns: tuple = (
        "Customer_Age", "Dependent_count", "Months_on_book",
        "Total_Relationship_Count", "Months_Inactive_12_mon",
        "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
        "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
        "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio",
    )
    categorical_columns: tuple = (
        "Gender", "Education_Level", "Marital_Status",
        "Income_Category", "Card_Category", "Cluster_ID",
    )

    # Risk bands, for a human readable label alongside the raw probability
    high_risk_threshold: float = 0.5
    medium_risk_threshold: float = 0.2


class PredictionPipeline:
    """ Takes a single new customer's raw feature values (as the FastAPI form/JSON would supply) and returns:
      1) Cluster_ID 
      2) Churn_Probability 
      3) Risk_Label (High / Medium / Low, for a readable UI result)
    """


    def __init__(self, config: PredictionPipelineConfig = PredictionPipelineConfig()):
        self.config = config
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            with open(self.config.kmeans_model_path, "rb") as f:
                self.kmeans = pickle.load(f)
            with open(self.config.segmentation_scaler_path, "rb") as f:
                self.segmentation_scaler = pickle.load(f)
            with open(self.config.classification_preprocessor_path, "rb") as f:
                self.classification_preprocessor = pickle.load(f)
            with open(self.config.classification_model_path, "rb") as f:
                self.classification_model = pickle.load(f)
            logging.info("Prediction pipeline artifacts loaded")
        except Exception as e:
            raise CustomException(e, sys)

    def _assign_cluster(self, input_df: pd.DataFrame) -> int:
        features = input_df[list(self.config.behavior_features)].copy()

        for col in self.config.skewed_features:
            features[col] = np.log1p(features[col])
        scaled = self.segmentation_scaler.transform(features)
        cluster_id = int(self.kmeans.predict(scaled)[0])
        return cluster_id

    def _predict_churn(self, input_df: pd.DataFrame, cluster_id: int) -> float:
        input_df = input_df.copy()
        input_df["Cluster_ID"] = str(cluster_id)

        feature_columns = list(self.config.numeric_columns) + list(self.config.categorical_columns)
        X = input_df[feature_columns]
        X_transformed = self.classification_preprocessor.transform(X)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        churn_probability = float(self.classification_model.predict_proba(X_transformed)[0, 1])
        return churn_probability

    def _risk_label(self, churn_probability: float) -> str:
        if churn_probability >= self.config.high_risk_threshold:
            return "High"
        elif churn_probability >= self.config.medium_risk_threshold:
            return "Medium"
        return "Low"

    def predict(self, customer_data: dict) -> dict:
        try:
            input_df = pd.DataFrame([customer_data])

            cluster_id = self._assign_cluster(input_df)
            churn_probability = self._predict_churn(input_df, cluster_id)
            risk_label = self._risk_label(churn_probability)

            result = {
                "Cluster_ID": cluster_id,
                "Churn_Probability": round(churn_probability, 4),
                "Risk_Label": risk_label,
            }
            logging.info(f"Prediction: {result}")
            return result

        except Exception as e:
            raise CustomException(e, sys)