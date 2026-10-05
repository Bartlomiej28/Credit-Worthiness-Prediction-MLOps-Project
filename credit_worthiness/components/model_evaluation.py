import os
import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score

from credit_worthiness.configuration.aws_connection import S3Client
from credit_worthiness.constants import TARGET_COLUMN
from credit_worthiness.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from credit_worthiness.entity.config_entity import ModelEvaluationConfig
from credit_worthiness.entity.estimator import TargetValueMapping
from credit_worthiness.entity.s3_estimator import CreditWorthinessEstimator
from credit_worthiness.exception import CreditWorthinessException
from credit_worthiness.logger import logging
from credit_worthiness.utils.main_utils import add_engineered_features, load_object


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def get_best_model(self) -> Optional[object]:
        """
        Method Name :   get_best_model
        Description :   This function returns the model currently in production. The S3
                        registry is used when AWS is configured, otherwise the local
                        saved_models directory is used as the registry.

        Output      :   Returns the production model object, or None when there is none yet
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if S3Client.is_configured():
                model_path = self.model_eval_config.s3_model_key_path
                estimator = CreditWorthinessEstimator(
                    bucket_name=self.model_eval_config.bucket_name, model_path=model_path
                )
                if estimator.is_model_present(model_path=model_path):
                    logging.info("Found a production model in the S3 model registry")
                    return estimator
                logging.info("No production model in the S3 model registry")
            else:
                logging.info("AWS is not configured, falling back to the local model registry")

            local_model_path = self.model_eval_config.local_model_file_path
            if os.path.exists(local_model_path):
                logging.info(f"Found a production model in the local registry: {local_model_path}")
                return load_object(file_path=local_model_path)

            logging.info("No production model found, the freshly trained model becomes the first one")
            return None
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function evaluates the trained model against the production
                        model and decides which one is the best

        Output      :   Returns the evaluation response
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df = add_engineered_features(test_df)

            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            y = y.map(TargetValueMapping()._asdict()).astype(int)

            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)

            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            difference = trained_model_f1_score - tmp_best_model_score

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=difference > self.model_eval_config.changed_threshold_score,
                difference=difference,
            )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function initiates all the steps of the model evaluation

        Output      :   Returns the model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e
