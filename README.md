# cm_predict
S2 full image prediction

# Dependencies
The following system dependencies are needed:
* conda 4.9 or later
* python 3.6 or later

## Setup
1. Create a conda environment.

        conda env create -f environment.yml

2. Copy `config/config_example.json` and adapt it to your needs.

## Usage

Cloudmask generating can be run as follows:

```
conda activate cm_predict
python cm_predict.py -c config/your_config.json
```
