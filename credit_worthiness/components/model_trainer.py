import sys
from typing import Tuple

import numpy as np
from neuro_mf import ModelFactory
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from credit_worthiness.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from credit_worthiness.entity.config_entity import ModelTrainerConfig
from credit_worthiness.entity.estimator import CreditWorthinessModel
from credit_worthiness.exception import CreditWorthinessException
from credit_worthiness.logger import logging
from credit_worthiness.utils.main_utils import load_numpy_array_data, load_object, save_object


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        """
        :param data_transformation_artifact: Output reference of the data transformation artifact stage
        :param model_trainer_config: Configuration for the model trainer
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, ClassificationMetricArtifact]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and its report

        Output      :   Returns the best model detail and the metric artifact object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_mf to get the best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)

            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(
                X=x_train, y=y_train, base_accuracy=self.model_trainer_config.expected_accuracy
            )
            model_obj = best_model_detail.best_model

            y_pred = model_obj.predict(x_test)

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1_score(y_test, y_pred),
                precision_score=precision_score(y_test, y_pred, zero_division=0),
                recall_score=recall_score(y_test, y_pred),
                accuracy_score=accuracy_score(y_test, y_pred),
                roc_auc_score=roc_auc_score(y_test, y_pred),
            )

            logging.info(f"Metrics of the best model on the test set: {metric_artifact}")

            return best_model_detail, metric_artifact

        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model trainer steps

        Output      :   Returns the model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with a score higher than the base score")
                raise Exception("No best model found with a score higher than the base score")

            credit_worthiness_model = CreditWorthinessModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model_detail.best_model,
            )
            logging.info("Created the creditworthiness model object with the preprocessor and the model")

            save_object(self.model_trainer_config.trained_model_file_path, credit_worthiness_model)
            logging.info("Saved the best model file")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e
