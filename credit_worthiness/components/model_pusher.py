import os
import shutil
import sys

from credit_worthiness.configuration.aws_connection import S3Client
from credit_worthiness.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from credit_worthiness.entity.config_entity import ModelPusherConfig
from credit_worthiness.entity.s3_estimator import CreditWorthinessEstimator
from credit_worthiness.exception import CreditWorthinessException
from credit_worthiness.logger import logging


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact, model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of the model evaluation artifact stage
        :param model_pusher_config: Configuration for the model pusher
        """
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

    def push_to_local_registry(self) -> str:
        """
        Method Name :   push_to_local_registry
        Description :   Copies the accepted model to the local registry, so that the
                        prediction pipeline can serve it without any cloud access

        Output      :   Path of the model inside the local registry
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            local_model_path = self.model_pusher_config.local_model_file_path
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            shutil.copy(self.model_evaluation_artifact.trained_model_path, local_model_path)
            logging.info(f"Saved the accepted model in the local registry: {local_model_path}")
            return local_model_path
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def push_to_s3_registry(self) -> str:
        """
        Method Name :   push_to_s3_registry
        Description :   Uploads the accepted model to the S3 model registry when AWS is configured

        Output      :   Key of the model inside the bucket, or an empty string when skipped
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if not S3Client.is_configured():
                logging.info("AWS is not configured, skipping the upload to the S3 model registry")
                return ""

            estimator = CreditWorthinessEstimator(
                bucket_name=self.model_pusher_config.bucket_name,
                model_path=self.model_pusher_config.s3_model_key_path,
            )
            estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
            logging.info(
                f"Uploaded the model to s3://{self.model_pusher_config.bucket_name}/"
                f"{self.model_pusher_config.s3_model_key_path}"
            )
            return self.model_pusher_config.s3_model_key_path
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This function initiates all the steps of the model pusher

        Output      :   Returns the model pusher artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            local_model_path = self.push_to_local_registry()
            s3_model_path = self.push_to_s3_registry()

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=s3_model_path,
                local_model_path=local_model_path,
            )

            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e
