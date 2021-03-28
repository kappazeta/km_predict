# cm_predict
S2 full image prediction

## Dependencies
The following system dependencies are needed:
* conda 4.9 or later
* python 3.6 or later

## Setup
1. Create a conda environment.

        conda env create -f environment.yml

2. Copy `config/config_example.json` and adapt it to your needs.
3. In order to run sub-tiling procedure cm_vsm should be installed (https://github.com/kappazeta/cm-vsm).

## Input data
In the root of repository create a ```/data``` folder and place .SAFE product to it 

## Usage

Cloudmask generating can be run as follows:

```
conda activate cm_predict
python cm_predict.py -c config/your_config.json
```

## Output
The predictor will generate sub-tiles masks under ```/prediction``` folder and full S2 mask under ```/big_image``` folder
