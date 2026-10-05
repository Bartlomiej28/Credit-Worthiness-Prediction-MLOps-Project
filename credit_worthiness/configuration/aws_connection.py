import os

import boto3

from credit_worthiness.constants import (
    AWS_ACCESS_KEY_ID_ENV_KEY,
    AWS_REGION_ENV_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_KEY,
    REGION_NAME,
)


class S3Client:

    s3_client = None
    s3_resource = None

    def __init__(self, region_name: str = None):
        """
        This class reads the AWS credentials from the environment variables and opens a
        connection to S3. It raises an exception when the environment variables are not set.
        """
        region_name = region_name or os.getenv(AWS_REGION_ENV_KEY) or REGION_NAME

        if S3Client.s3_resource is None or S3Client.s3_client is None:
            __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)
            if __access_key_id is None:
                raise Exception(f"Environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY} is not set.")
            if __secret_access_key is None:
                raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set.")

            S3Client.s3_resource = boto3.resource(
                "s3",
                aws_access_key_id=__access_key_id,
                aws_secret_access_key=__secret_access_key,
                region_name=region_name,
            )
            S3Client.s3_client = boto3.client(
                "s3",
                aws_access_key_id=__access_key_id,
                aws_secret_access_key=__secret_access_key,
                region_name=region_name,
            )
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client

    @staticmethod
    def is_configured() -> bool:
        """
        Tells whether the AWS credentials are available, so that the training and the
        prediction pipelines can fall back to the local model registry instead of failing.
        """
        return (
            os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY) is not None
            and os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY) is not None
        )
