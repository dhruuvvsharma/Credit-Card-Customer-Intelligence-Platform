import sys

from src.my_project.exceptions import CustomException
from src.my_project.logger import logging

from src.my_project.components.data_ingestion import DataIngestion
from src.my_project.components.segmentation import CustomerSegmentation
from src.my_project.components.data_transformation import DataTransformation
from src.my_project.components.model_trainer import ModelTrainer
from src.my_project.components.retention_analysis import RetentionAnalysis


class TrainingPipeline:
    """Full flow, in order:
      1. Data Ingestion    : raw.csv (full data) + train.csv/test.csv (stratified split)
      2. Segmentation      : KMeans on raw.csv (unsupervised, needs every customer)---> cluster_labels.csv (CLIENTNUM + Cluster_ID)
      3. Transformation     : merges Cluster_ID into train/test via CLIENTNUM join,encodes/scales features ---> train_arr/test_arr
      4. Model Training     : RandomForest + XGBoost, best model by recall -> classification_model.pkl
      5. Retention Analysis : scores every customer, merges Cluster_ID + Churn_Probability + actual Attrition_Flag ---> retention_dataset.csv + cluster_summary.json

      Segmentation runs on the full raw dataset, independently of the train/test split
    """

    def run(self):
        logging.info("--- Training pipeline started ---")
        try:
            # Step 1: Ingestion
            ingestion = DataIngestion()
            raw_path, train_path, test_path = ingestion.initiate_data_ingestion()
            logging.info(f"(1/5) Ingestion done, raw: {raw_path}, train: {train_path}, test: {test_path}")

            # Step 2: Segmentation (unsupervised)
            segmentation = CustomerSegmentation()
            cluster_labels_path, silhouette = segmentation.initiate_segmentation()
            logging.info(f"(2/5) Segmentation done, silhouette: {silhouette:.4f}, labels: {cluster_labels_path}")

            # Step 3: Transformation (merges Cluster_ID into train/test)
            transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation()
            logging.info(f"(3/5) Transformation done, train_arr: {train_arr.shape}, test_arr: {test_arr.shape}")

            # Step 4: Model training (classification)
            trainer = ModelTrainer()
            model_path, best_name, metrics = trainer.initiate_model_training(train_arr, test_arr)
            logging.info(f"(4/5) Training done, best model: {best_name}, metrics: {metrics}")

            # Step 5: Retention analysis (business insight layer)
            retention = RetentionAnalysis()
            retention_dataset_path, cluster_summary_path, cluster_summary = retention.initiate_retention_analysis()
            logging.info(
                f"(5/5) Retention analysis done, highest risk: "
                f"{cluster_summary['_highest_risk_cluster']}, lowest risk: {cluster_summary['_lowest_risk_cluster']}"
            )

            logging.info("--- Training pipeline completed successfully ---")

            return {
                "raw_path": raw_path,
                "train_path": train_path,
                "test_path": test_path,
                "cluster_labels_path": cluster_labels_path,
                "silhouette_score": silhouette,
                "preprocessor_path": preprocessor_path,
                "model_path": model_path,
                "best_model": best_name,
                "metrics": metrics,
                "retention_dataset_path": retention_dataset_path,
                "cluster_summary_path": cluster_summary_path,
                "cluster_summary": cluster_summary,
            }

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    results = pipeline.run()

    print("\n--- Training Pipeline Results ---")
    for key, value in results.items():
        if key == "cluster_summary":
            continue  # printed separately below because it's a nested dict
        print(f"{key}: {value}")

    print("\n--- Cluster Summary ---")
    summary = results["cluster_summary"]
    print(f"Overall churn rate: {summary['_overall_churn_rate']:.2%}")
    print(f"Highest risk cluster: {summary['_highest_risk_cluster']}")
    print(f"Lowest risk cluster:  {summary['_lowest_risk_cluster']}")
    for cluster_id, stats in summary.items():
        if cluster_id.startswith("_"):
            continue

        print(f"  Cluster {cluster_id}: {stats['customer_count']} customers, "
              f"churn rate {stats['actual_churn_rate']:.2%} "
              f"({stats['churn_rate_vs_overall']:+.2%} vs overall)")