import sys

import numpy as np
import pandas as pd

from credit_worthiness.constants import (
    RAW_DATA_FILE_PATH,
    RAW_DATA_SHEET_NAME,
    SCHEMA_FILE_PATH,
)
from credit_worthiness.exception import CreditWorthinessException
from credit_worthiness.logger import logging
from credit_worthiness.utils.main_utils import read_yaml_file


class CreditData:
    """
    This class exports the raw CreditWorthiness workbook as a clean pandas DataFrame.
    """

    def __init__(self, file_path: str = RAW_DATA_FILE_PATH, sheet_name: str = RAW_DATA_SHEET_NAME):
        try:
            self.file_path = file_path
            self.sheet_name = sheet_name
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e

    @staticmethod
    def _normalise_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        The workbook ships category labels with leading, trailing and doubled
        spaces ("non resident either unemployed or  unskilled "), which would
        otherwise become distinct categories of the encoders.
        """
        for column in dataframe.select_dtypes(include="object").columns:
            dataframe[column] = (
                dataframe[column].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            )
        return dataframe

    def export_data_as_dataframe(self) -> pd.DataFrame:
        """
        Method Name :   export_data_as_dataframe
        Description :   This method reads the source workbook, renames the abbreviated
                        columns to their descriptive names and normalises the categories

        Output      :   the whole dataset is returned as a pandas DataFrame
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Reading the dataset from {self.file_path} (sheet: {self.sheet_name})")

            dataframe = pd.read_excel(self.file_path, sheet_name=self.sheet_name)

            dataframe = dataframe.rename(columns=self._schema_config["raw_column_mapping"])
            dataframe = self._normalise_text_columns(dataframe)
            dataframe = dataframe.replace({"na": np.nan, "NA": np.nan, "": np.nan})
            dataframe = dataframe.drop_duplicates().reset_index(drop=True)

            logging.info(f"Exported dataframe with shape: {dataframe.shape}")

            return dataframe
        except Exception as e:
            raise CreditWorthinessException(e, sys) from e
