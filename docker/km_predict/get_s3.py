import os
import argparse
from pathlib import Path
from configparser import ConfigParser
from urllib.error import HTTPError
from sentinelhub import SHConfig
from sentinelhub.aws import AwsProductRequest
from sentinelhub.exceptions import DownloadFailedException
from copernicus_s3_utils import download_product_by_id
from os import environ
import sys

sh_config = None
path_aws_credentials = Path("~/.aws/credentials").expanduser()
if path_aws_credentials.exists():
    sh_config = SHConfig()
    cfg_parser = ConfigParser()
    cfg_parser.read(path_aws_credentials)
    sh_config.aws_access_key_id = cfg_parser["default"]["aws_access_key_id"]
    sh_config.aws_secret_access_key = cfg_parser["default"]["aws_secret_access_key"]

try:
    parser = argparse.ArgumentParser("get_s3")
    parser.add_argument(
        "product", help="Name of the Sentinel-2 product to download from S3", type=str
    )
    parser.add_argument("dest", help="Path to download to", type=str)
    args = parser.parse_args()

    product_request = AwsProductRequest(
        product_id=args.product,
        data_folder=args.dest,
        safe_format=True,
        config=sh_config,
    )
    dataset = product_request.save_data()

except DownloadFailedException as e:
    if "404 Client Error: Not Found for url" in str(e):
        print("entered exception")
        REQUIRED_ENV_VARS = {"COPERNICUS_ACCESS_KEY_ID", "COPERNICUS_SECRET_ACCESS_KEY", "COPERNICUS_OAUTH_ACCESS_KEY_ID", "COPERNICUS_OAUTH_SECRET_ACCESS_KEY"}
        missing_variables  = REQUIRED_ENV_VARS.difference(environ)
        if len(missing_variables) > 0:
            raise EnvironmentError(f'Failed because {missing_variables} are not set')
            sys.exit(1)
        else:
            downloaded_files = download_product_by_id(args.product, args.dest)
