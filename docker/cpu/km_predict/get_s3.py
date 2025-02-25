import os
import argparse
from pathlib import Path
from configparser import ConfigParser
from sentinelhub import SHConfig
from sentinelhub.aws import AwsProductRequest

sh_config = None
path_aws_credentials = Path("~/.aws/credentials").expanduser()
if path_aws_credentials.exists():
    sh_config = SHConfig()

    cfg_parser = ConfigParser()
    cfg_parser.read(path_aws_credentials)
    sh_config.aws_access_key_id = cfg_parser["default"]["aws_access_key_id"]
    sh_config.aws_secret_access_key = cfg_parser["default"]["aws_secret_access_key"]

parser = argparse.ArgumentParser("get_s3")
parser.add_argument("product", help="Name of the Sentinel-2 product to download from S3", type=str)
parser.add_argument("dest", help="Path to download to", type=str)
args = parser.parse_args()

product_request = AwsProductRequest(
    product_id=args.product,
    data_folder=args.dest,
    safe_format=True,
    config=sh_config
)
dataset = product_request.save_data()
