from datetime import datetime, timezone
from glob import glob
import logging
from urllib.parse import urlparse
from urllib.error import HTTPError
from sentinelhub.exceptions import DownloadFailedException
import boto3
from pathlib import Path
import os
import re
import time
from typing import List, Union
from sentinelhub.config import SHConfig
from sentinelhub.api.catalog import SentinelHubCatalog
from sentinelhub.data_collections import DataCollection


from botocore.exceptions import ClientError
from botocore.config import Config


import requests

# Copernicus S3 Bucket
COPERNICUS_ACCESS_KEY_ID = os.environ.get("COPERNICUS_ACCESS_KEY_ID")
COPERNICUS_SECRET_ACCESS_KEY = os.environ.get("COPERNICUS_SECRET_ACCESS_KEY")
# Copernicus OAUTH
COPERNICUS_OAUTH_ACCESS_KEY_ID = os.environ.get("COPERNICUS_OAUTH_ACCESS_KEY_ID")
COPERNICUS_OAUTH_SECRET_ACCESS_KEY = os.environ.get(
    "COPERNICUS_OAUTH_SECRET_ACCESS_KEY"
)


CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"

MAX_RETRIES = 20
MAX_WAIT_TIME = 120


retry_config = Config(
    retries={
        "max_attempts": 20,  # Maximum number of attempts
        "mode": "standard",  # Retry mode, can be 'standard' or 'adaptive'
    }
)


client = boto3.client(
    "s3",
    aws_access_key_id=COPERNICUS_ACCESS_KEY_ID,
    aws_secret_access_key=COPERNICUS_SECRET_ACCESS_KEY,
    endpoint_url="https://eodata.dataspace.copernicus.eu",
    config=retry_config,
)


def build_catalog_config() -> SHConfig:
    """
    Build Config for sentinelhub.Catalog API to query.
    Returns SHConfig object.
    """

    config = SHConfig()
    config.sh_client_id = COPERNICUS_OAUTH_ACCESS_KEY_ID
    config.sh_client_secret = COPERNICUS_OAUTH_SECRET_ACCESS_KEY
    config.sh_token_url = CDSE_TOKEN_URL
    config.sh_base_url = CDSE_BASE_URL
    config.save("cdse")
    return config


def get_file_folders(s3_client, bucket_name, prefix=""):
    file_names = []
    folders = []

    default_kwargs = {"Bucket": bucket_name, "Prefix": prefix}
    next_token = ""

    while next_token is not None:
        updated_kwargs = default_kwargs.copy()
        if next_token != "":
            updated_kwargs["ContinuationToken"] = next_token
        response = s3_client.list_objects_v2(**updated_kwargs)
        contents = response.get("Contents")

        for result in contents:
            key = result.get("Key")
            if key[-1] == "/":
                folders.append(key)
            else:
                file_names.append(key)
        next_token = response.get("NextContinuationToken")
    return file_names, folders


def download_files(
    s3_client,
    bucket_name: str,
    local_path: str,
    files: List,
    folders: List,
    product_id: str,
    include_files: Union[List, None] = None,
):

    for folder in folders:
        __folder = Path(folder).name
        folder_path = os.path.join(local_path, __folder)
        # Create all folders in the path
        Path(folder_path).mkdir(parents=True, exist_ok=True)

    local_file_paths_list = []

    for file_name in files:
        if include_files is None or Path(file_name).name in include_files:
            """
            local_file_path = os.path.join(
                local_path, Path(file_name).parent.name, Path(file_name).name
            )
            """
            local_file_path = (
                local_path
                + product_id
                + ".SAFE"
                + file_name.split(product_id + ".SAFE")[1]
            )
            local_folder_path = os.path.dirname(local_file_path)
            os.makedirs(local_folder_path, exist_ok=True)
            download_file_with_custom_backoff(
                s3_client, bucket_name, file_name, local_file_path
            )
            local_file_paths_list.append(local_file_path)
    return local_file_paths_list


def download_file_with_custom_backoff(s3_client, bucket_name, object_key, file_path):
    """
    Downl`oad files with BAckoff syst`em to allow 429 HTTP code to retrigger the call

    BAckoff is by 1.5 times and max wait time is set to 120 seconds. Max retries is 20 times.
    """
    retries = 0
    max_retries = MAX_RETRIES
    max_wait_time = MAX_WAIT_TIME
    backoff_factor = 1.5  # Base exponential factor for backoff

    while retries < max_retries:
        try:
            print("downloading :" + str(object_key) + "to " + str(file_path))
            s3_client.download_file(bucket_name, object_key, file_path)
            return
        except ClientError as e:
            if e.response["Error"]["Code"] in [
                "429",
                "500",
                "503",
            ]:  # Handle 429 and server errors
                retries += 1
                wait_time = min(max_wait_time, backoff_factor**retries)
                logging.info(
                    f"Rate limited or server error. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to download {object_key}: {e}")
                break
    else:
        logging.error(f"Failed to download {object_key} after {max_retries} retries")


def get_s3_location(product_id: str) -> Union[str, None]:
    """Queries Copernicus using product_id to get the S3 location of the product.
    Product_id should not include the file extension .SAFE at the end."""
    print("enetered get_s3_location")
    config = build_catalog_config()
    catalog = SentinelHubCatalog(config=config)
    query = f"{product_id}.SAFE"
    try:
        s3_location = catalog.get_feature(DataCollection.SENTINEL2_L2A, query)[
            "assets"
        ]["data"]["href"]
        print("s3 location is " + str(s3_location))
        return s3_location
    except DownloadFailedException as err:
        return None


def download_product_by_id(
    product_id: str, target_path: str, include_files: Union[List, None] = None
):
    """
    Download product from s3-like API of copernicus. Product_id is first used to resolve the s3 download path. Target_path is where to download the files

    """
    product_path = get_s3_location(product_id)
    if product_path == None:
        logging.info(
            f"Product path (s3 location) not found for {product_id} in Copernicus. Exiting."
        )
        return None
    parsed_url = urlparse(product_path)
    bucket_name = parsed_url.netloc
    object_path = parsed_url.path.strip("/")
    files, folders = get_file_folders(client, bucket_name, object_path)
    return download_files(
        client, bucket_name, target_path, files, folders, product_id, include_files
    )
