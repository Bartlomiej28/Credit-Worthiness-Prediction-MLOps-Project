import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from credit_worthiness.constants import TARGET_COLUMN
from credit_worthiness.data_access.credit_data import CreditData
from credit_worthiness.entity.artifact_entity import DataIngestionArtifact
from credit_worthiness.entity.config_entity import DataIngestionConfig
from credit_worthiness.exception import CreditWorthinessException
from credit_worthiness.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports the source workbook into a csv file

        Output      :   data is returned as an artifact of the data ingestion component
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Exporting data from the source workbook")
            credit_data = CreditData(
                file_path=self.data_ingestion_config.source_file_path,
                sheet_name=self.data_ingestion_config.source_sheet_name,
            )
            dataframe = credit_data.export_data_as_dataframe()
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into a train set and a test set
                        based on the split ratio, keeping the class balance of both sets

        Output      :   train.csv and test.csv are written to the ingested directory
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of DataIngestion class")

        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=self.data_ingestion_config.random_state,
                stratify=dataframe[TARGET_COLUMN],
            )
            logging.info("Performed train test split on the dataframe")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Exported train and test file path.")
            logging.info("Exited split_data_as_train_test method of DataIngestion class")
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion component of the training pipeline

        Output      :   train set and test set are returned as the artifacts of the component
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            dataframe = self.export_data_into_feature_store()

            logging.info("Got the data from the source workbook")

            self.split_data_as_train_test(dataframe)

            logging.info("Performed train test split on the dataset")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            logging.info("Exited initiate_data_ingestion method of DataIngestion class")
            return data_ingestion_artifact
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e
