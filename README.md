# cm_predict
Sentinel-2 full image prediction that support Level-1C and Level-2A input products

## Dependencies
The following system dependencies are needed:
* micromamba 0.18 or later (https://github.com/mamba-org/mamba)
* python 3.6 or later
* cm_vsm dependencies

Due to the long environment solve times with Miniconda, we have switched to Micromamba. If you're still using Conda, Miniconda or similar, simply substitute `micromamba` with `conda` in the relevant commands below.

## Setup
1. Create a micromamba environment.

        micromamba env create -f environment.yml

2. Copy `config/config_example.json` and adapt it to your needs.
3. In order to run sub-tiling procedure cm_vsm should be installed (https://github.com/kappazeta/cm-vsm).
4. Make sure that your `GDAL_DATA` environment variable has been set, according to your GDAL version instead of the placeholder `YOUR_GDAL_VERSION` below:

        GDAL_DATA=/usr/share/gdal/YOUR_GDAL_VERSION


## Input data
In the root of repository create a ```/data``` folder and copy or symlink the .SAFE product into it.

## Usage
Cloudmask inference can be run as follows:

    micromamba activate cm_predict
    python cm_predict.py -c config/your_config.json

It is possible to overwrite product_name in config file with command line argument -product

    python cm_predict.py -c config/your_config.json -product S2B_MSIL2A_20200401T093029_N0214_R136_T34UFA_20200401T122148

If the prediction for the same product is running multiple times and .CVAT folder is created under ```/data``` folder, it might be convenient to disable sub_tiling procedure for the next run by -t

    python cm_predict.py -c config/your_config.json -product S2B_MSIL2A_20200401T093029_N0214_R136_T34UFA_20200401T122148 -t

## Output
The predictor will generate sub-tiles masks under ```/prediction``` folder and full S2 mask under ```/big_image``` folder

## Troubleshooting
Potential solutions for typical issues encountered during setup or usage.

### Unable to open EPSG support file
Sentinel-2 product splitting fails with the following messages:

    INFO: CMP.P: Extracting geo-coordinates.
    ERROR 4: Unable to open EPSG support file gcs.csv.  Try setting the GDAL_DATA environment variable to point to the directory containing EPSG csv files.
    ERROR 4: Unable to open EPSG support file gcs.csv.  Try setting the GDAL_DATA environment variable to point to the directory containing EPSG csv files.
    INFO: CMP.P: Projection:
    terminate called after throwing an instance of 'INFO: CMP.P: Projecting AOI polygon into pixel coordinates.
    GDALOGRException'
      what():  GDAL OGR error : Failed to import spatial reference from EPSG, Generic failure
    Magick: abort due to signal 6 (SIGABRT) "Abort"...

This indicates that the environment variable `GDAL_DATA` has not been configured correctly. This could be done in a variety of ways and the preferred method depends on your linux distribution. An export call for the variable (for example, `GDAL_DATA=/usr/share/gdal/2.2`) could be added to your `.bashrc`, `.profile`, etc. Alternatively, the variable could be set together with the python call, for example:

    GDAL_DATA=/usr/share/gdal/2.2 python cm_predict.py -c config/your_config.json

### Filesystem error
Sentinel-2 product splitting fails with the following messages:

    terminate called after throwing an instance of 'std::filesystem::__cxx11::filesystem_error'
      what():  filesystem error: directory iterator cannot open directory: No such file or directory [YOUR_DIRECTORY/cm_predict/data/S2B_MSIL1C_20200401T093029_N0209_R136_T34UFA_20200401T113334.SAFE.SAFE/GRANULE/]
    Magick: abort due to signal 6 (SIGABRT) "Abort"...

This means that cm_predict cannot find the directory with the `product_name` specified in the configuration file. The product name in the configuration file should be provided without the `.SAFE` suffix.
