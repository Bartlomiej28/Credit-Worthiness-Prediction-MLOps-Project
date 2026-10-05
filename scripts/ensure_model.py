import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credit_worthiness.configuration.aws_connection import S3Client
from credit_worthiness.entity.config_entity import CreditWorthinessPredictorConfig
from credit_worthiness.entity.s3_estimator import CreditWorthinessEstimator
from credit_worthiness.pipeline.training_pipeline import TrainPipeline


def registry_bucket_exists(bucket_name: str) -> bool:
    """
    Tells whether the configured model registry bucket is reachable.
    """
    try:
        S3Client().s3_client.head_bucket(Bucket=bucket_name)
        return True
    except Exception as e:
        print(f"The model registry bucket is not reachable: {e}", file=sys.stderr)
        return False


def model_present_in_registry() -> bool:
    """
    Tells whether the S3 model registry already serves a model.
    """
    config = CreditWorthinessPredictorConfig()
    estimator = CreditWorthinessEstimator(
        bucket_name=config.model_bucket_name,
        model_path=config.model_file_path,
    )
    return estimator.is_model_present(model_path=config.model_file_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="retrain even when the registry already holds a model",
    )
    args = parser.parse_args()

    config = CreditWorthinessPredictorConfig()

    if not S3Client.is_configured():
        print("AWS is not configured, the model cannot be published to the S3 registry.", file=sys.stderr)
        return 1

    if not registry_bucket_exists(config.model_bucket_name):
        print(
            f"Create the bucket first:\n"
            f"    aws s3 mb s3://{config.model_bucket_name}\n"
            f"or point MODEL_BUCKET_NAME in credit_worthiness/constants/__init__.py at an existing one.",
            file=sys.stderr,
        )
        return 1

    if not args.force and model_present_in_registry():
        print(f"The S3 registry already serves s3://{config.model_bucket_name}/{config.model_file_path}.")
        return 0

    print("Training a model and publishing it to the S3 registry ...")
    TrainPipeline().run_pipeline()

    if not model_present_in_registry():
        print(
            "The training run finished but the registry still holds no model. "
            "The candidate was most likely rejected by the model evaluation step.",
            file=sys.stderr,
        )
        return 1

    print(f"Published s3://{config.model_bucket_name}/{config.model_file_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
