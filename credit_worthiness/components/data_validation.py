import json
import os
import sys

import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset
from pandas import DataFrame

from credit_worthiness.constants import SCHEMA_FILE_PATH
from credit_worthiness.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from credit_worthiness.entity.config_entity import DataValidationConfig
from credit_worthiness.exception import CreditWorthinessException
from credit_worthiness.logger import logging
from credit_worthiness.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of the data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns

        Output      :   Returns a bool value based on the validation result
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is the required number of columns present: [{status}]")
            return status
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of the numerical and categorical columns

        Output      :   Returns a bool value based on the validation result
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return not (missing_categorical_columns or missing_numerical_columns)
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    @staticmethod
    def has_no_missing_values(df: DataFrame) -> bool:
        """
        Method Name :   has_no_missing_values
        Description :   The source dataset is complete, so any missing value means
                        that the incoming data does not follow the expected contract

        Output      :   Returns a bool value based on the validation result
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                logging.info(f"Columns with missing values: {missing.to_dict()}")
                return False
            return True
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    @staticmethod
    def _summarise_drift(json_report: dict) -> dict:
        """
        Translates the Evidently snapshot into a compact, serialisable summary.
        """
        columns = {}
        n_drifted_features = 0
        drift_share = 0.0
        drift_share_threshold = 0.5

        for metric in json_report.get("metrics", []):
            config = metric.get("config", {})
            metric_type = str(config.get("type", ""))
            value = metric.get("value")

            if metric_type.endswith("DriftedColumnsCount"):
                n_drifted_features = int(value["count"])
                drift_share = float(value["share"])
                drift_share_threshold = float(config.get("drift_share", 0.5))
            elif metric_type.endswith("ValueDrift"):
                method = str(config.get("method", ""))
                threshold = float(config.get("threshold", 0.05))
                score = float(value)
                drifted = score < threshold if "p_value" in method else score > threshold
                columns[str(config.get("column"))] = {
                    "method": method,
                    "score": score,
                    "threshold": threshold,
                    "drift_detected": bool(drifted),
                }

        return {
            "n_features": len(columns),
            "n_drifted_features": n_drifted_features,
            "drift_share": drift_share,
            "drift_share_threshold": drift_share_threshold,
            "dataset_drift": bool(drift_share >= drift_share_threshold),
            "columns": columns,
        }

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Method Name :   detect_dataset_drift
        Description :   This method validates whether drift is detected between two datasets

        Output      :   Returns a bool value based on the validation result
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            report = Report(metrics=[DataDriftPreset()])
            snapshot = report.run(
                reference_data=Dataset.from_pandas(reference_df),
                current_data=Dataset.from_pandas(current_df),
            )

            summary = self._summarise_drift(json.loads(snapshot.json()))

            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=summary)

            os.makedirs(os.path.dirname(self.data_validation_config.drift_report_html_file_path), exist_ok=True)
            snapshot.save_html(self.data_validation_config.drift_report_html_file_path)

            logging.info(f"{summary['n_drifted_features']}/{summary['n_features']} drift detected.")

            return summary["dataset_drift"]
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component of the pipeline

        Output      :   Returns the data validation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")

            train_df, test_df = (
                DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path),
            )

            for name, dataframe in (("training", train_df), ("test", test_df)):
                if not self.validate_number_of_columns(dataframe=dataframe):
                    validation_error_msg += f"Columns are missing in the {name} dataframe. "
                if not self.is_column_exist(df=dataframe):
                    validation_error_msg += f"Required columns are missing in the {name} dataframe. "
                if not self.has_no_missing_values(df=dataframe):
                    validation_error_msg += f"Missing values found in the {name} dataframe. "

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                validation_error_msg = "Drift detected" if drift_status else "Drift not detected"
                logging.info(validation_error_msg)
            else:
                logging.info(f"Validation error: {validation_error_msg}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e
