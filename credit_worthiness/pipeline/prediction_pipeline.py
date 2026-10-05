import os
import sys

import numpy as np
from pandas import DataFrame

from credit_worthiness.configuration.aws_connection import S3Client
from credit_worthiness.entity.config_entity import CreditWorthinessPredictorConfig
from credit_worthiness.entity.estimator import TargetValueMapping
from credit_worthiness.entity.s3_estimator import CreditWorthinessEstimator
from credit_worthiness.exception import CreditWorthinessException
from credit_worthiness.logger import logging
from credit_worthiness.utils.main_utils import add_engineered_features, load_object

NUMERIC_INPUT_FEATURES = [
    "credit_duration",
    "credit_amount",
    "installment_rate",
    "age",
    "existing_credits",
    "num_dependents",
]


class CreditWorthinessData:
    def __init__(
        self,
        checking_balance,
        credit_duration,
        credit_amount,
        credit_history,
        credit_purpose,
        savings_balance,
        employment_duration,
        installment_rate,
        marital_status_gender,
        other_parties,
        residence_duration,
        property_type,
        age,
        installment_plans,
        housing_type,
        existing_credits,
        job_type,
        num_dependents,
        foreign_worker,
    ):
        """
        CreditWorthiness data constructor
        Input: all the features the trained model needs for a prediction
        """
        try:
            self.checking_balance = checking_balance
            self.credit_duration = credit_duration
            self.credit_amount = credit_amount
            self.credit_history = credit_history
            self.credit_purpose = credit_purpose
            self.savings_balance = savings_balance
            self.employment_duration = employment_duration
            self.installment_rate = installment_rate
            self.marital_status_gender = marital_status_gender
            self.other_parties = other_parties
            self.residence_duration = residence_duration
            self.property_type = property_type
            self.age = age
            self.installment_plans = installment_plans
            self.housing_type = housing_type
            self.existing_credits = existing_credits
            self.job_type = job_type
            self.num_dependents = num_dependents
            self.foreign_worker = foreign_worker

        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def get_input_data_as_dict(self) -> dict:
        """
        This function returns a dictionary built from the CreditWorthinessData input
        """
        logging.info("Entered get_input_data_as_dict method of CreditWorthinessData class")

        try:
            input_data = {
                "checking_balance": [self.checking_balance],
                "credit_duration": [self.credit_duration],
                "credit_amount": [self.credit_amount],
                "credit_history": [self.credit_history],
                "credit_purpose": [self.credit_purpose],
                "savings_balance": [self.savings_balance],
                "employment_duration": [self.employment_duration],
                "installment_rate": [self.installment_rate],
                "marital_status_gender": [self.marital_status_gender],
                "other_parties": [self.other_parties],
                "residence_duration": [self.residence_duration],
                "property_type": [self.property_type],
                "age": [self.age],
                "installment_plans": [self.installment_plans],
                "housing_type": [self.housing_type],
                "existing_credits": [self.existing_credits],
                "job_type": [self.job_type],
                "num_dependents": [self.num_dependents],
                "foreign_worker": [self.foreign_worker],
            }

            logging.info("Created the creditworthiness input data dict")
            logging.info("Exited get_input_data_as_dict method of CreditWorthinessData class")

            return input_data

        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def get_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame built from the CreditWorthinessData input.
        The values coming from an HTML form are strings, so the numeric features are
        cast before the derived features are computed.
        """
        try:
            dataframe = DataFrame(self.get_input_data_as_dict())
            for feature in NUMERIC_INPUT_FEATURES:
                dataframe[feature] = dataframe[feature].astype(float)
            return add_engineered_features(dataframe)

        except Exception as e:
            raise CreditWorthinessException(e, sys) from e


class CreditWorthinessClassifier:
    def __init__(self, prediction_pipeline_config: CreditWorthinessPredictorConfig = CreditWorthinessPredictorConfig()):
        """
        :param prediction_pipeline_config: Configuration used to locate the production model
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            self.model = None
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def get_model(self) -> object:
        """
        Loads the production model. The S3 registry has priority; when AWS is not
        configured, or the bucket holds no model yet, the local registry written by
        the model pusher is used instead.
        """
        try:
            if self.model is not None:
                return self.model

            if S3Client.is_configured():
                model_path = self.prediction_pipeline_config.model_file_path
                estimator = CreditWorthinessEstimator(
                    bucket_name=self.prediction_pipeline_config.model_bucket_name,
                    model_path=model_path,
                )
                if estimator.is_model_present(model_path=model_path):
                    logging.info("Serving the model stored in the S3 model registry")
                    self.model = estimator
                    return self.model
                logging.info("No model in the S3 model registry, falling back to the local one")

            local_model_path = self.prediction_pipeline_config.local_model_file_path
            if os.path.exists(local_model_path):
                logging.info(f"Serving the model stored in the local registry: {local_model_path}")
                self.model = load_object(file_path=local_model_path)
                return self.model

            raise Exception(
                "No trained model is available. Run the training pipeline first "
                "(python demo.py or the /train endpoint)."
            )
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        """
        Returns the predicted class for every row of the given dataframe
        """
        try:
            logging.info("Entered predict method of CreditWorthinessClassifier class")
            return self.get_model().predict(dataframe)

        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def predict_with_probability(self, dataframe: DataFrame) -> tuple:
        """
        Returns the predicted label and, when the estimator supports it, the
        probability that the applicant turns out to be a bad credit risk.
        """
        try:
            prediction = int(self.predict(dataframe)[0])
            label = TargetValueMapping().reverse_mapping()[prediction]

            probability = None
            try:
                probability = float(self.get_model().predict_proba(dataframe)[0][TargetValueMapping().bad])
            except Exception as e:
                logging.info(f"The selected estimator does not expose probabilities: {e}")

            return label, probability

        except Exception as e:
            raise CreditWorthinessException(e, sys) from e
