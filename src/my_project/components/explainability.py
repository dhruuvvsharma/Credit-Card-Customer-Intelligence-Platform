import os
import sys
import pickle
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from src.my_project.exceptions import CustomException
from src.my_project.logger import logging


@dataclass
class ExplainabilityConfig:
    model_path: str = os.path.join("artifacts", "classification_model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "classification_preprocessor.pkl")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    cluster_labels_path: str = os.path.join("artifacts", "cluster_labels.csv")

    shap_values_path: str = os.path.join("artifacts", "shap_values.pkl")
    feature_importance_path: str = os.path.join("artifacts", "metrics", "feature_importance.json")

    id_column: str = "CLIENTNUM"
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

class ModelExplainability:
    """Explains the trained classifier's predictions using SHAP's TreeExplainer(it works directly on XGBoost/RandomForest 
    without needing a background sample unlike KernelExplainer which would be far slower on ~2000 test rows)"""

    def __init__(self, config: ExplainabilityConfig = ExplainabilityConfig()):
        self.config = config

    def load_inputs(self):
        with open(self.config.model_path, "rb") as f:
            model = pickle.load(f)
        with open(self.config.preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

        test_df = pd.read_csv(self.config.test_data_path)
        cluster_labels = pd.read_csv(self.config.cluster_labels_path)
        test_df = test_df.merge(cluster_labels, on=self.config.id_column, how="left")
        test_df["Cluster_ID"] = test_df["Cluster_ID"].astype("Int64").astype(str)

        return model, preprocessor, test_df

    def get_feature_names(self, preprocessor) -> list:

        return list(preprocessor.get_feature_names_out())


    def compute_shap_values(self, model, preprocessor, test_df):
        feature_columns = list(self.config.numeric_columns) + list(self.config.categorical_columns)
        X_test = test_df[feature_columns]
        X_transformed = preprocessor.transform(X_test)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        feature_names = self.get_feature_names(preprocessor)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        return shap_values, X_transformed, feature_names

    def build_feature_importance(self, shap_values, feature_names) -> dict:
        """Mean absolute SHAP value per feature — the standard global importance
        ranking, readable without needing to load the full pickle to inspect."""
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = dict(zip(feature_names, mean_abs_shap.tolist()))
        ranked = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
        return {k: round(v, 5) for k, v in ranked.items()}




    def initiate_explainability(self):
        logging.info("Explainability (SHAP) started")
        try:
            model, preprocessor, test_df = self.load_inputs()
            shap_values, X_transformed, feature_names = self.compute_shap_values(
                model, preprocessor, test_df
            )

            os.makedirs(os.path.dirname(self.config.shap_values_path), exist_ok=True)
            with open(self.config.shap_values_path, "wb") as f:
                pickle.dump(
                    {
                        "shap_values": shap_values,
                        "X_transformed": X_transformed,
                        "feature_names": feature_names,
                    },
                    f,
                )
            logging.info(f"SHAP values saved: {self.config.shap_values_path}")

            feature_importance = self.build_feature_importance(shap_values, feature_names)
            os.makedirs(os.path.dirname(self.config.feature_importance_path), exist_ok=True)
            with open(self.config.feature_importance_path, "w") as f:
                json.dump(feature_importance, f, indent=2)
            logging.info(f"Feature importance saved: {self.config.feature_importance_path}")

            top_5 = list(feature_importance.items())[:5]
            logging.info(f"Top 5 features by mean |SHAP|: {top_5}")

            return self.config.shap_values_path, feature_importance

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    explainability = ModelExplainability()
    shap_path, feature_importance = explainability.initiate_explainability()

    print(f"SHAP values saved to: {shap_path}")
    print("\nTop 10 features by mean |SHAP| (global importance):")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
        print(f"  {i}. {feature}: {importance}")