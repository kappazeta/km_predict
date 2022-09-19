#!/bin/bash

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
else
    input_product_level=L1C
fi

cat >"${path_config}" <<EOF
{
  "cm_vsm": {
    "path": "cm_vsm",
    "env": {
      "LD_LIBRARY_PATH": "."
    }
  },
  "folder_name": "data",
  "product_name": "${input_product}",
  "level_product": "${input_product_level}",
  "overlapping": 0.0625,
  "tile_size": 512,
  "resampling_method" : "sinc",
  "architecture": "DeepLabv3Plus",
  "batch_size": 1
}
EOF

function py_exec() {
    cd /home/km_predict/ && source /etc/profile.d/conda.sh && conda activate km_predict && python3 $@
}

echo "Activating Mamba environment"

echo "Downloading ${input_product}"
py_exec /home/get_s3.py ${input_product} /home/km_predict/data/
echo "Running km_predict"
py_exec km_predict.py -c "${path_config}" ${@:3}
echo "Compressing the output"
gdal_translate -co COMPRESS=LZMA -co TILED=YES /home/km_predict/prediction/${input_product}/${input_product_short}.tif /home/km_predict/prediction/${input_product}/${input_product_short}.compressed.tif
echo "Creating overviews"
gdaladdo /home/km_predict/prediction/${input_product}/${input_product_short}.compressed.tif 2 4 8 16 32 64 128 256
echo "Uploading results to S3"
aws s3 cp --no-progress /home/km_predict/prediction/${input_product}/${input_product_short}.compressed.tif ${dir_path_out}${input_product}/${input_product_short}.tif
