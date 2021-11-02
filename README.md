# cm_predict
Sentinel-2 full image prediction that support Level-1C and Level-2A input products

## Dependencies
The following system dependencies are needed:
* conda 4.9 or later
* python 3.6 or later
* cm_vsm dependencies

## Setup
1. Create a conda environment.

        conda env create -f environment.yml

2. Copy `config/config_example.json` and adapt it to your needs.
3. In order to run sub-tiling procedure cm_vsm should be installed (https://github.com/kappazeta/cm-vsm).
4. Make sure that your `GDAL_DATA` environment variable has been set, according to your GDAL version instead of the placeholder `YOUR_GDAL_VERSION` below:

        GDAL_DATA=/usr/share/gdal/YOUR_GDAL_VERSION


## Input data
In the root of repository create a ```/data``` folder and place .SAFE product to it 

## Usage
Cloudmask inference can be run as follows:

```
conda activate cm_predict
python cm_predict.py -c config/your_config.json
```

It is possible to overwrite product_name in config file with command line argument -product
```
python cm_predict.py -c config/your_config.json -product S2B_MSIL2A_20200401T093029_N0214_R136_T34UFA_20200401T122148
```
If the prediction for the same product is running multiple times and .CVAT folder is created under ```/data``` folder, it might be convenient to disable sub_tiling procedure for the next run by -t
```
python cm_predict.py -c config/your_config.json -product S2B_MSIL2A_20200401T093029_N0214_R136_T34UFA_20200401T122148 -t
```

## Output
The predictor will generate sub-tiles masks under ```/prediction``` folder and full S2 mask under ```/big_image``` folder
