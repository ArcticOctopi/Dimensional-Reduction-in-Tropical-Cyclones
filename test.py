import boto3
from botocore import UNSIGNED
from botocore.client import Config
from os import listdir
from os.path import isfile, join
#Test
def list_s3_files(BUCKET_NAME, prefix=''):

    """
    Lists files in an AWS S3 bucket, optionally filtered by a prefix.

    Args:
        bucket_name (str): The name of the S3 bucket.
        prefix (str, optional): An optional prefix to filter files (e.g., 'folder/').
    """
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    bucket_name = BUCKET_NAME
    # Handle pagination for more than 1000 objects
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    file_list = []
    for page in pages:
        print()
        if 'Contents' in page:
            for obj in page['Contents']:
                file_list.append(obj)
        else:
            print(f"No files found in '{bucket_name}' with prefix '{prefix}'.")
    return file_list

def get_links():

    link_folder = 'Data/links'
    loaded_list = []
    
    if does_file_exist_in_dir(link_folder):
        
        with open('Data/links/link_list.txt', 'r') as f:
            for line in f:
                loaded_list.append(line.strip()) # .strip() removes newline characters
        return loaded_list
    else:
        file_list = list_s3_files(BUCKET_NAME=BUCKET_NAME)

        with open('Data/links/link_list.txt', 'w') as f:
            for item in file_list:
                 if (('f024' in item['Key'] or 'f027' in item['Key']) 
                      and 'idx' not in item['Key']
                      and 'storm.atm' in item['Key']):
                        loaded_list.append(item['Key'])
                        f.write(item['Key'] + '\n')
        return loaded_list
    
def does_file_exist_in_dir(path):
        return any(isfile(join(path, i)) for i in listdir(path))

CMP_MAX = 990.0
BUCKET_NAME = 'noaa-nws-hafs-pds'

frame_links = get_links()