import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler

from credit_worthiness.constants import SCHEMA_FILE_PATH, TARGET_COLUMN
from credit_worthiness.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from credit_worthiness.entity.config_entity import DataTransformationConfig
from credit_worthiness.entity.estimator import TargetValueMapping
from credit_worthiness.exception import CreditWorthinessException
from credit_worthiness.logger import logging
from credit_worthiness.utils.main_utils import (
    add_engineered_features,
    drop_columns,
    read_yaml_file,
    save_numpy_array_data,
    save_object,
)


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        """
        :param data_ingestion_artifact: Output reference of the data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        :param data_validation_artifact: Output reference of the data validation artifact stage
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data

        Output      :   the data transformer object is created and returned
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            oh_columns = self._schema_config["oh_columns"]
            or_columns = self._schema_config["or_columns"]
            transform_columns = self._schema_config["transform_columns"]
            num_features = self._schema_config["num_features"]
            ordinal_categories = self._schema_config["ordinal_categories"]

            logging.info("Got the transformation columns from the schema config")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown="ignore")
            ordinal_encoder = OrdinalEncoder(
                categories=[ordinal_categories[column] for column in or_columns],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )

            logging.info("Initialized StandardScaler, OneHotEncoder and OrdinalEncoder")

            transform_pipe = Pipeline(steps=[("transformer", PowerTransformer(method="yeo-johnson"))])

            logging.info("Initialized PowerTransformer")

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features),
                ]
            )

            logging.info("Created the preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")

            return preprocessor

        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component of the pipeline

        Output      :   the data transformer steps are performed and the preprocessor object is created
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            drop_cols = self._schema_config["drop_columns"]
            target_value_mapping = TargetValueMapping()._asdict()

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            logging.info("Got the train features and the target of the training dataset")

            input_feature_train_df = add_engineered_features(input_feature_train_df)
            logging.info("Added the monthly_installment column to the training dataset")

            input_feature_train_df = drop_columns(df=input_feature_train_df, cols=drop_cols)
            logging.info("Dropped the columns listed in drop_columns of the training dataset")

            target_feature_train_df = target_feature_train_df.map(target_value_mapping).astype(int)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            input_feature_test_df = add_engineered_features(input_feature_test_df)
            logging.info("Added the monthly_installment column to the test dataset")

            input_feature_test_df = drop_columns(df=input_feature_test_df, cols=drop_cols)
            logging.info("Dropped the columns listed in drop_columns of the test dataset")

            target_feature_test_df = target_feature_test_df.map(target_value_mapping).astype(int)

            logging.info("Got the test features and the target of the test dataset")
            logging.info("Applying the preprocessing object on the training and test dataframes")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Used the preprocessor object to fit transform the train features")

            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Used the preprocessor object to transform the test features")

            logging.info("Applying SMOTE on the training dataset")

            smt = SMOTE(random_state=42)

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )

            logging.info("Applied SMOTE on the training dataset")
            logging.info("The test set is left untouched so that it keeps the real class balance")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Created the train array and the test array")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info("Saved the preprocessor object and the transformed arrays")
            logging.info("Exited initiate_data_transformation method of DataTransformation class")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

        except Exception as e:
            raise CreditWorthinessException(e, sys) from e
