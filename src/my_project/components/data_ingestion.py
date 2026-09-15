import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.my_project.exceptions import CustomException
from src.my_project.logger import logging


@dataclass
class DataIngestionConfig:
    # Source: the raw CSV as downloaded, not touched by any pipeline stage
    source_data_path: str = os.path.join("data", "raw", "BankChurners.csv")

    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

    test_size: float = 0.2
    random_state: int = 42
    target_column: str = "Attrition_Flag"


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            df = pd.read_csv(self.config.source_data_path)
            logging.info(f"Read source dataset, shape: {df.shape}")

            # Drop Naive Bayes leakage columns if present 
            leak_cols = [c for c in df.columns if "Naive_Bayes" in c]
            if leak_cols:
                df = df.drop(columns=leak_cols)
                logging.info(f"Dropped leakage columns: {leak_cols}")

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # raw.csv is untouched dataset. Segmentation runs on this,
            # since KMeans is unsupervised and needs every customer, not a train subset
            df.to_csv(self.config.raw_data_path, index=False)
            logging.info(f"Raw data saved to {self.config.raw_data_path}")

            # train/test split — only used by classification (Stage 2)
            train_set, test_set = train_test_split(df,test_size=self.config.test_size,random_state=self.config.random_state,stratify=df[self.config.target_column],)

            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)
            logging.info(f"Train/test split saved; train: {train_set.shape}, test: {test_set.shape}")

            logging.info("Data ingestion completed")

            return (self.config.raw_data_path,self.config.train_data_path,self.config.test_data_path,)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    raw_path, train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"Raw:   {raw_path}")
    print(f"Train: {train_path}")
    print(f"Test:  {test_path}")