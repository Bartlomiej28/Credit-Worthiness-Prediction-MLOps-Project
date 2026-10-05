import os
import sys

import dill
import numpy as np
import yaml
from pandas import DataFrame

from credit_worthiness.exception import CreditWorthinessException
from credit_worthiness.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CreditWorthinessException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file, sort_keys=False)
    except Exception as e:
        raise CreditWorthinessException(e, sys) from e


def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")

    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of utils")

        return obj

    except Exception as e:
        raise CreditWorthinessException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise CreditWorthinessException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CreditWorthinessException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CreditWorthinessException(e, sys) from e


def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    drop the columns from a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns method of utils")

    try:
        df = df.drop(columns=[col for col in cols if col in df.columns], axis=1)

        logging.info("Exited the drop_columns method of utils")

        return df
    except Exception as e:
        raise CreditWorthinessException(e, sys) from e


def add_engineered_features(df: DataFrame) -> DataFrame:
    """
    Add the features derived from the raw ones.

    monthly_installment expresses how much of the credit has to be repaid every
    month and combines the requested amount with the requested duration, which
    turned out to be far more informative than either column on its own.

    df: pandas DataFrame holding at least credit_amount and credit_duration
    return: a copy of the DataFrame with the derived columns appended
    """
    logging.info("Entered add_engineered_features method of utils")

    try:
        df = df.copy()
        df["credit_amount"] = df["credit_amount"].astype(float)
        df["credit_duration"] = df["credit_duration"].astype(float)
        df["monthly_installment"] = df["credit_amount"] / df["credit_duration"]

        logging.info("Exited the add_engineered_features method of utils")

        return df
    except Exception as e:
        raise CreditWorthinessException(e, sys) from e
