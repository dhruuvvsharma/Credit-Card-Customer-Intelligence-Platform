import os
import sys
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from src.my_project.exceptions import CustomException
from src.my_project.logger import logging

@dataclass
class PredictionPipelineConfig:
    kmeans_model_path: str = os.path.join("artifacts", "kmeans_model.pkl")
    segmentation_scaler_path: str = os.path.join("artifacts", "scaler.pkl")
    classification_preprocessor_path: str = os.path.join("artifacts", "classification_preprocessor.pkl")
    classification_model_path: str = os.path.join("artifacts", "classification_model.pkl")

    # Same 5 locked features segmentation was trained on
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

    # Risk bands for a human readable label alongside the raw probability
    high_risk_threshold: float = 0.5
    medium_risk_threshold: float = 0.2

    # Raw feature name---> human readable label, for the SHAP output.
    # Categorical dummies (e.g. "categorical__Gender_M") 
    readable_names: dict = None

    def __post_init__(self):
        self.readable_names = {
            "numeric__Customer_Age": "Age",
            "numeric__Dependent_count": "Number of dependents",
            "numeric__Months_on_book": "Tenure with bank",
            "numeric__Total_Relationship_Count": "Products held",
            "numeric__Months_Inactive_12_mon": "Months inactive (last year)",
            "numeric__Contacts_Count_12_mon": "Support contacts (last year)",
            "numeric__Credit_Limit": "Credit limit",
            "numeric__Total_Revolving_Bal": "Revolving balance",
            "numeric__Avg_Open_To_Buy": "Available credit",
            "numeric__Total_Amt_Chng_Q4_Q1": "Spending change (Q4 vs Q1)",
            "numeric__Total_Trans_Amt": "Total transaction amount",
            "numeric__Total_Trans_Ct": "Transaction count",
            "numeric__Total_Ct_Chng_Q4_Q1": "Transaction count change (Q4 vs Q1)",
            "numeric__Avg_Utilization_Ratio": "Card utilization",
        }


class PredictionPipeline:
    """
    Takes a single new customer's raw feature values (as the FastAPI form/JSON
    would supply) and returns:
      1) Cluster_ID (which segment they fall into)
      2) Churn_Probability (from the trained classifier)
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


            self.explainer = shap.TreeExplainer(self.classification_model)
            self.feature_names = list(self.classification_preprocessor.get_feature_names_out())

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

    def _predict_churn(self, input_df: pd.DataFrame, cluster_id: int):
        input_df = input_df.copy()
        input_df["Cluster_ID"] = str(cluster_id)

        feature_columns = list(self.config.numeric_columns) + list(self.config.categorical_columns)
        X = input_df[feature_columns]
        X_transformed = self.classification_preprocessor.transform(X)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        churn_probability = float(self.classification_model.predict_proba(X_transformed)[0, 1])
        return churn_probability, X_transformed

    def _readable_label(self, raw_name: str) -> str:
        if raw_name in self.config.readable_names:
            return self.config.readable_names[raw_name]
        # Categorical dummy, e.g. "categorical__Gender_M" or "categorical__Cluster_ID_2"
        if raw_name.startswith("categorical__"):
            stripped = raw_name.replace("categorical__", "")
            return stripped.replace("_", " ")
        return raw_name

    def _explain_prediction(self, X_transformed) -> list:
        """Top 3 features driving this specific prediction, by SHAP value"""

        shap_values = self.explainer.shap_values(X_transformed)
        row = shap_values[0]

        top_idx = np.argsort(np.abs(row))[::-1][:4]
        max_impact = max(abs(row[i]) for i in top_idx) if len(top_idx) else 1.0

        factors = []
        for i in top_idx:
            impact = float(abs(row[i]))
            factors.append({
                "feature": self._readable_label(self.feature_names[i]),
                "direction": "increases" if row[i] > 0 else "decreases",
                "impact": round(impact, 3),
                "relative_pct": round(float((impact / max_impact) * 100), 1) if max_impact else 0.0,
            })
        return factors

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
            churn_probability, X_transformed = self._predict_churn(input_df, cluster_id)
            risk_label = self._risk_label(churn_probability)
            key_factors = self._explain_prediction(X_transformed)

            result = {
                "Cluster_ID": cluster_id,
                "Churn_Probability": round(churn_probability, 4),
                "Risk_Label": risk_label,
                "Key_Factors": key_factors,
            }
            logging.info(f"Prediction: {result}")
            return result

        except Exception as e:
            raise CustomException(e, sys)