import sys

import numpy as np
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from credit_worthiness.exception import CreditWorthinessException
from credit_worthiness.logger import logging


class TargetValueMapping:
    """
    Encodes the target column: an applicant whose credit score is bad is the
    positive class (1), because that is the outcome the bank wants to detect.
    """

    def __init__(self):
        self.good: int = 0
        self.bad: int = 1

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class CreditWorthinessModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocessor
        :param trained_model_object: Input Object of trained model
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        """
        Function accepts raw inputs and then transforms the raw input using the
        preprocessing_object, which guarantees that the inputs are in the same
        format as the training data. At last it performs prediction on the
        transformed features.
        """
        logging.info("Entered predict method of CreditWorthinessModel class")

        try:
            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def predict_proba(self, dataframe: DataFrame) -> np.ndarray:
        """
        Returns the probability of every class for the given raw inputs.
        Raises an AttributeError when the underlying estimator cannot score.
        """
        logging.info("Entered predict_proba method of CreditWorthinessModel class")

        try:
            transformed_feature = self.preprocessing_object.transform(dataframe)
            return self.trained_model_object.predict_proba(transformed_feature)

        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
