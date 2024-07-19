#!/bin/bash

trap 'echo "Cancelled by user"; exit' INT

if [ "$#" -lt 2 ]; then
    echo "KappaMask S3 processor"
    echo "Usage: km_s3 PRODUCT_NAME OUTPUT_S3_PATH [OPTIONS]"
    echo "  PRODUCT_NAME - Sentinel-2 product name. For example, S2A_MSIL2A_20200509T094041_N0214_R036_T35VME_20200509T111504"
    echo "  OUTPUT_S3_PATH - S3 path to store the results in. For example, s3://kappamask/km_pred_out/"
    echo "  OPTIONS - Additional arguments for km_predict. For example, -cpu, -g"
    exit 1
fi

wd=/data
input_product=$1
input_product_short=$(echo "$1" | sed 's@S2[AB]\+_MSI\(L[12AC]\+\)_\([0-9T]\+\)_N[0-9]\+_R[0-9]\+_\(T[0-9A-Z]\+\)_[0-9T]\+@\1_\3_\2_KZ_10m@')
path_config=/home/km_predict/config/config.json

# Ensure that the output path has a slash at the end.
re='.*/$'
dir_path_out=$2
[[ ${dir_path_out} =~ ${re} ]] || dir_path_out+='/'

# L1C or L2A product?
if [[ "${input_product}" =~ "MSIL2A" ]]; then
    input_product_level=L2A
    bands="AOT,B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12,WVP"
else
    input_product_level=L1C
    bands="B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12"
fi

cat >"${path_config}" <<EOF
{
  "cm_vsm": {
    "path": "cm_vsm",
    "env": {
      "LD_LIBRARY_PATH": "."
    }
  },
  "folder_name": "${wd}",
  "product_name": "${input_product}",
  "level_product": "${input_product_level}",
  "overlapping": 0.0625,
  "tile_size": 512,
  "resampling_method" : "sinc",
  "architecture": "DeepLabv3Plus",
  "batch_size": 1
}
EOF


function config_aws() {
    aws configure set region ${AWS_REGION}
    aws configure set aws_access_key_id ${AWS_ACCESS_KEY}
    aws configure set aws_secret_access_key ${AWS_SECRET_KEY}
}

function process() {
    echo "Downloading ${input_product}"
    python3 /home/get_s3.py ${input_product} ${wd}/
    if [ $? -eq 0 ]
    then 
    echo "Splitting ${input_product}"
    cm_vsm -d "${wd}/${input_product}.SAFE" -j -1 -b "${bands}" -S 512 -f 0 -m sinc -o 0.0625
    echo "Running km_predict"
    cd /home/km_predict && python3 km_predict.py -c "${path_config}" -t ${@:3}
    mkdir -p ${wd}/prediction
    mv /home/km_predict/prediction/${input_product} ${wd}/prediction/
    echo "Compressing the output"
    gdal_translate -co COMPRESS=LZMA -co TILED=YES ${wd}/prediction/${input_product}/${input_product_short}.tif ${wd}/prediction/${input_product}/${input_product_short}.compressed.tif
    echo "Creating overviews"
    gdaladdo ${wd}/prediction/${input_product}/${input_product_short}.compressed.tif 2 4 8 16 32 64 128 256
    echo "Uploading results to S3"
    rm ${wd}/prediction/${input_product}/${input_product_short}.tif
    mv ${wd}/prediction/${input_product}/${input_product_short}.compressed.tif ${wd}/prediction/${input_product}/${input_product_short}.tif
    aws s3 cp --no-progress --recursive ${wd}/prediction/${input_product}/ ${dir_path_out}${input_product}/
    else
    echo "python checks failed.Exiting."
    fi
       
}


if [[ -v AWS_REGION && -v AWS_ACCESS_KEY && -v AWS_SECRET_KEY ]]; then
    config_aws
    process
else
    echo "Expecting the following environment variables to be defined:"
    echo "  AWS_REGION, AWS_ACCESS_KEY, AWS_SECRET_KEY"
    exit 1
fi