import os

PIPELINE_NAME: str = "creditworthiness"
ARTIFACT_DIR: str = "artifact"

RAW_DATA_FILE_PATH: str = os.path.join("data", "CreditWorthiness.xlsx")
RAW_DATA_SHEET_NAME: str = "Data"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
FILE_NAME: str = "creditworthiness.csv"

MODEL_FILE_NAME: str = "model.pkl"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

TARGET_COLUMN: str = "credit_score"
SCHEMA_FILE_PATH: str = os.path.join("config", "schema.yaml")
FORM_FILE_PATH: str = os.path.join("config", "form.yaml")

SAVED_MODEL_DIR: str = "saved_models"

AWS_ACCESS_KEY_ID_ENV_KEY: str = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY: str = "AWS_SECRET_ACCESS_KEY"
AWS_REGION_ENV_KEY: str = "AWS_DEFAULT_REGION"
REGION_NAME: str = "us-east-1"


"""
Data Ingestion related constants start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTION_RANDOM_STATE: int = 42


"""
Data Validation related constants start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_DRIFT_REPORT_HTML_FILE_NAME: str = "report.html"


"""
Data Transformation related constants start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


"""
Model Trainer related constants start with MODEL_TRAINER VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")


"""
Model Evaluation and Model Pusher related constants
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME: str = "creditworthiness-model-registry"
MODEL_PUSHER_S3_KEY: str = "model-registry"


APP_HOST: str = "0.0.0.0"
APP_PORT: int = 8080
