import os
import sys
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.my_project.exceptions import CustomException
from src.my_project.logger import logging


@dataclass
class DataTransformationConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    cluster_labels_path: str = os.path.join("artifacts", "cluster_labels.csv")
    preprocessor_path: str = os.path.join("artifacts", "classification_preprocessor.pkl")

    id_column: str = "CLIENTNUM"
    target_column: str = "Attrition_Flag"

    # Cluster_ID is a generated categorical label not the numeric ones
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


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

    def get_preprocessor(self) -> ColumnTransformer:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, list(self.config.numeric_columns)),
            ("categorical", categorical_pipeline, list(self.config.categorical_columns)),
        ])

        return preprocessor

    def merge_cluster_id(self, df: pd.DataFrame, cluster_labels: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(cluster_labels, on=self.config.id_column, how="left")

        if merged["Cluster_ID"].isnull().any():
            missing = merged["Cluster_ID"].isnull().sum()
            logging.info(f"Warning: {missing} rows had no matching Cluster_ID after merge")

        # Cluster_ID is categorical, so OneHotEncoder treats it as such
        merged["Cluster_ID"] = merged["Cluster_ID"].astype("Int64").astype(str)
        return merged

    def initiate_data_transformation(self):
        logging.info("Data transformation started")
        try:
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)
            cluster_labels = pd.read_csv(self.config.cluster_labels_path)

            train_df = self.merge_cluster_id(train_df, cluster_labels)
            test_df = self.merge_cluster_id(test_df, cluster_labels)
            logging.info("Cluster_ID merged into train and test")

            # Target encode to binary: churned (left) = 1 , retained (Existing) = 0
            target_map = {"Attrited Customer": 1, "Existing Customer": 0}
            y_train = train_df[self.config.target_column].map(target_map)
            y_test = test_df[self.config.target_column].map(target_map)

            feature_columns = list(self.config.numeric_columns) + list(self.config.categorical_columns)
            X_train = train_df[feature_columns]
            X_test = test_df[feature_columns]

            preprocessor = self.get_preprocessor()
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # handling both sparse and dense output depending on OneHotEncoder version
            if hasattr(X_train_arr, "toarray"):
                X_train_arr = X_train_arr.toarray()
                X_test_arr = X_test_arr.toarray()

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)
            with open(self.config.preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logging.info(
                f"Data transformation completed, train_arr: {train_arr.shape}, "
                f"test_arr: {test_arr.shape}"
            )

            return train_arr, test_arr, self.config.preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation()
    print(f"Train array shape: {train_arr.shape}")
    print(f"Test array shape: {test_arr.shape}")
    print(f"Preprocessor saved to: {preprocessor_path}")