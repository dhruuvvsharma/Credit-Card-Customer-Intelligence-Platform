import os
import sys
import pickle
import json
from dataclasses import dataclass

import pandas as pd

from src.my_project.exceptions import CustomException
from src.my_project.logger import logging


@dataclass
class RetentionAnalysisConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    cluster_labels_path: str = os.path.join("artifacts", "cluster_labels.csv")
    preprocessor_path: str = os.path.join("artifacts", "classification_preprocessor.pkl")
    model_path: str = os.path.join("artifacts", "classification_model.pkl")

    # Final PowerBI ready output
    retention_dataset_path: str = os.path.join("artifacts", "retention_dataset.csv")

    # Cluster-level rollup the actual business insight
    cluster_summary_path: str = os.path.join("artifacts", "cluster_summary.json")


    id_column: str = "CLIENTNUM"
    target_column: str = "Attrition_Flag"

    categorical_columns: tuple = (
        "Gender", "Education_Level", "Marital_Status",
        "Income_Category", "Card_Category", "Cluster_ID",
    )
    numeric_columns: tuple = (
        "Customer_Age", "Dependent_count", "Months_on_book",
        "Total_Relationship_Count", "Months_Inactive_12_mon",
        "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
        "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
        "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio",
    )

    # the same behavior features used for segmentation is used again here to describe *why* a cluster churns
    behavior_features: tuple = (
        "Total_Revolving_Bal", "Avg_Utilization_Ratio", "Total_Trans_Ct",
        "Total_Trans_Amt", "Contacts_Count_12_mon",
    )


class RetentionAnalysis:
    def __init__(self, config: RetentionAnalysisConfig = RetentionAnalysisConfig()):
        self.config = config

    def load_inputs(self):
        df = pd.read_csv(self.config.raw_data_path)
        cluster_labels = pd.read_csv(self.config.cluster_labels_path)

        with open(self.config.preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)
        with open(self.config.model_path, "rb") as f:
            model = pickle.load(f)

        return df, cluster_labels, preprocessor, model

    def score_all_customers(self, df, cluster_labels, preprocessor, model) -> pd.DataFrame:
        """Score every customer,this stage is for business reporting, not model evaluation, so using 
        full population is correct here rather than a leakage concern."""

        merged = df.merge(cluster_labels, on=self.config.id_column, how="left")
        merged["Cluster_ID"] = merged["Cluster_ID"].astype("Int64").astype(str)

        feature_columns = list(self.config.numeric_columns) + list(self.config.categorical_columns)
        X = merged[feature_columns]
        X_transformed = preprocessor.transform(X)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        merged["Churn_Probability"] = model.predict_proba(X_transformed)[:, 1]
        merged["Actual_Churned"] = (merged[self.config.target_column] == "Attrited Customer").astype(int)

        return merged

    def build_cluster_summary(self, scored_df: pd.DataFrame) -> dict:
        summary = {}
        overall_churn_rate = scored_df["Actual_Churned"].mean()

        for cluster_id, group in scored_df.groupby("Cluster_ID"):
            churn_rate = group["Actual_Churned"].mean()
            summary[str(cluster_id)] = {
                "customer_count": int(len(group)),
                "actual_churn_rate": round(float(churn_rate), 4),
                "avg_predicted_churn_probability": round(float(group["Churn_Probability"].mean()), 4),
                "churn_rate_vs_overall": round(float(churn_rate - overall_churn_rate), 4),
                "avg_behavior_features": {
                    col: round(float(group[col].mean()), 2)
                    for col in self.config.behavior_features
                },
            }

        # Rank clusters by risk, highest churn rate first 
        ranked = sorted(summary.items(), key=lambda kv: kv[1]["actual_churn_rate"], reverse=True)
        summary["_overall_churn_rate"] = round(float(overall_churn_rate), 4)
        summary["_highest_risk_cluster"] = ranked[0][0]
        summary["_lowest_risk_cluster"] = ranked[-1][0]

        return summary


    def initiate_retention_analysis(self):
        logging.info("Retention analysis started")
        try:
            df, cluster_labels, preprocessor, model = self.load_inputs()
            scored_df = self.score_all_customers(df, cluster_labels, preprocessor, model)

            os.makedirs(os.path.dirname(self.config.retention_dataset_path), exist_ok=True)
            scored_df.to_csv(self.config.retention_dataset_path, index=False)
            logging.info(f"Retention dataset saved: {scored_df.shape} -> {self.config.retention_dataset_path}")

            cluster_summary = self.build_cluster_summary(scored_df)
            with open(self.config.cluster_summary_path, "w") as f:
                json.dump(cluster_summary, f, indent=2)
            logging.info(
                f"Cluster summary saved — highest risk: {cluster_summary['_highest_risk_cluster']}, "
                f"lowest risk: {cluster_summary['_lowest_risk_cluster']}"
            )

            logging.info("Retention analysis completed")
            return self.config.retention_dataset_path, self.config.cluster_summary_path, cluster_summary

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    analysis = RetentionAnalysis()
    dataset_path, summary_path, summary = analysis.initiate_retention_analysis()

    print(f"Retention dataset: {dataset_path}")
    print(f"Cluster summary:   {summary_path}")
    print(f"\nOverall churn rate: {summary['_overall_churn_rate']:.2%}")
    print(f"Highest risk cluster: {summary['_highest_risk_cluster']}")
    print(f"Lowest risk cluster:  {summary['_lowest_risk_cluster']}")
    print()
    for cluster_id, stats in summary.items():
        if cluster_id.startswith("_"):
            continue
        print(f"Cluster {cluster_id}: {stats['customer_count']} customers, "
              f"churn rate {stats['actual_churn_rate']:.2%} "
              f"({stats['churn_rate_vs_overall']:+.2%} vs overall)")