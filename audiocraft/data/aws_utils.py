import boto3
from io import BytesIO
import os

def fetch_from_s3(bucket_name: str, file_key: str) -> BytesIO:
    """
    Fetch file fom S3 and return as a byte stream
    """
    s3 = boto3.client('s3', 
                      aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'), 
                      aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY'))
    
    object = s3.get_object(Bucket=bucket_name, Key=file_key)
    return BytesIO(object['Body'].read())