import os
import sys
import json
import pickle
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from xgboost import XGBClassifier

from src.my_project.exceptions import CustomException
from src.my_project.logger import logging


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "classification_model.pkl")
    classification_report_path: str = os.path.join("artifacts", "metrics", "classification_report.json")
    random_state: int = 42


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    def get_candidate_models(self):
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                random_state=self.config.random_state,
                class_weight="balanced",   # dataset is 84/16 imbalanced
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300,
                random_state=self.config.random_state,
                eval_metric="logloss",
                # scale_pos_weight approximates class_weight="balanced" for XGBoost
                scale_pos_weight=(1 - 0.1607) / 0.1607,
            ),
        }

    def evaluate_model(self, model, X_test, y_test):
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }
        report = classification_report(y_test, preds, output_dict=True)
        return metrics, report

    

    def initiate_model_training(self, train_arr, test_arr):
        logging.info("Model training started")
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            candidates = self.get_candidate_models()
            results = {}

            for name, model in candidates.items():
                model.fit(X_train, y_train)
                metrics, report = self.evaluate_model(model, X_test, y_test)
                results[name] = {"model": model, "metrics": metrics, "report": report}

                logging.info(f"{name} —-> {metrics}")


            # Select best model by recall on the churn class 
            # business context, missing an actual churner (false negative) costs more
            # than a false alarm, so recall matters more than raw accuracy here
            best_name = max(results, key=lambda n: results[n]["metrics"]["recall"])
            best_model = results[best_name]["model"]
            best_metrics = results[best_name]["metrics"]
            best_report = results[best_name]["report"]

            logging.info(f"Best model selected: {best_name} —-> {best_metrics}")

            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            with open(self.config.model_path, "wb") as f:
                pickle.dump(best_model, f)

            os.makedirs(os.path.dirname(self.config.classification_report_path), exist_ok=True)
            with open(self.config.classification_report_path, "w") as f:
                json.dump(
                    {
                        "best_model": best_name,
                        "metrics": best_metrics,
                        "report": best_report,
                        "all_candidates": {n: r["metrics"] for n, r in results.items()},
                    },
                    f,
                    indent=2,
                )

            logging.info("Model training completed")
            return self.config.model_path, best_name, best_metrics

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.my_project.components.data_transformation import DataTransformation

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation()

    trainer = ModelTrainer()
    model_path, best_name, metrics = trainer.initiate_model_training(train_arr, test_arr)

    print(f"Best model: {best_name}")
    print(f"Metrics: {metrics}")
    print(f"Saved to: {model_path}")