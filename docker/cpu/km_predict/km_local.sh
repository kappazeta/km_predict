#!/bin/bash

trap 'echo "Cancelled by user"; exit' INT

if [ "$#" -lt 1 ]; then
   echo "Local KappaMask processor"
   echo "Usage: km_local PRODUCT_NAME [OPTIONS]"
   echo "  PRODUCT_NAME - Sentinel-2 product name. For example, S2A_MSIL2A_20200509T094041_N0214_R036_T35VME_20200509T111504"
   echo "  OPTIONS - Additional arguments for km_predict. For example, -cpu, -g"
   exit 1
fi

wd=/data
input=$(basename ${1%.zip})
output=$(echo "$input" | sed 's@S2[ABC]\+_MSI\(L[12AC]\+\)_\([0-9T]\+\)_N[0-9]\+_R[0-9]\+_\(T[0-9A-Z]\+\)_[0-9T]\+@\1_\3_\2_KZ_10m@')
config=km_predict-config.json

# L1C or L2A product?
if [[ "${input}" =~ "MSIL2A" ]]; then
   input_product_level=L2A
   bands="AOT,B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12,WVP"
else
   input_product_level=L1C
   bands="B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12"
fi

echo "Changing working directory to ${wd}"
cd ${wd}

# Generate config file.
cat > "${wd}/${config}" <<EOF
{
 "cm_vsm": {
   "path": "cm_vsm",
   "env": {
     "LD_LIBRARY_PATH": "."
   }
 },
 "folder_name": "${wd}",
 "product_name": "${input}",
 "level_product": "${input_product_level}",
 "overlapping": 0.0625,
 "tile_size": 512,
 "resampling_method" : "sinc",
 "architecture": "DeepLabv3Plus",
 "batch_size": 1
}
EOF

# Attempt to unzip an archive if the .SAFE directory does not exist yet.
if [ ! -e "${wd}/${input}.SAFE" ]; then
   echo "Could not find a .SAFE directory, decompressing ${input}.zip"
   unzip -d ${wd} ${wd}/${input}.zip
fi

# Split the S2 product into subtiles.
echo "Splitting ${input}"
cm_vsm -d "${wd}/${input}.SAFE" -j -1 -b "${bands}" -S 512 -f 0 -m sinc -o 0.0625

# Run inference.
echo "Running km_predict"
python3 /home/km_predict/km_predict.py -c "${wd}/${config}" -l ${wd}/km_predict.log -t ${@:2}

# Postprocess the output.
echo "Compressing the output"
gdal_translate -co COMPRESS=LZMA -co TILED=YES ${wd}/prediction/${input}/${output}.tif ${wd}/${output}.tif
echo "Creating overviews"
gdaladdo ${wd}/${output}.tif 2 4 8 16 32 64 128 256

echo "KAPPAMASK_OUTPUT_PRODUCT ${output}.tif"
