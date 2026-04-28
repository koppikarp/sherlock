import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Initialize the S3 client for public access (no credentials needed)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket_name = 'czi-subcell-public'
prefix = 'hpa-processed/'

# Fetch the list of objects
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

if 'Contents' in response:
    for obj in response['Contents']:
        print(obj['Key'])
else:
    print("No files found or bucket/prefix is incorrect.")