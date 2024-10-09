import boto3
import os
import pandas as pd
from botocore.exceptions import ClientError
import pymupdf  
import uuid




class s3Handler:
    def __init__(self):
        self._s3_resource_handler = boto3.resource("s3")
        self._s3_client = boto3.client("s3")

    def list_buckets(self):
        """
        Get the buckets in all Regions for the current account.

        :param s3_resource: A Boto3 S3 resource. This is a high-level resource in Boto3
                            that contains collections and factory methods to create
                            other high-level S3 sub-resources.
        :return: The list of buckets.
        """
        try:
            buckets = list(self._s3_resource_handler.buckets.all())
            print(f"buckets: {buckets}")
        except ClientError:
            raise
        else:
            return buckets

    def download_file(
        self, bucket_name, object_path, download_path, local_filename
    ):
        try:
            if not os.path.exists(download_path):
                os.makedirs(download_path)

            self._s3_client.download_file(
                bucket_name,
                f"{object_path}",
                f"{download_path}/{local_filename}",
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print("The object does not exist.")
            else:
                raise

    def upload_file(self, local_file, bucket_name, s3_filename):
        try:
            self._s3_client.upload_file(local_file, bucket_name, s3_filename, {"Metadata": {"Content-Type": "application/pdf"}})
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print("The object does not exist.")
            else:
                raise