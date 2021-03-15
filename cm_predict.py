import json
import argparse

p=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

p.add_argument("-c", "--config", action="store", dest="path_config",help="Path to the configuration file.")

args=p.parse_args()

with open(args.path_config, "rt") as fi:
    cfg = json.load(fi)

def load_config(path):
    with open(path, "rt") as fi:
        d = json.load(fi)
    data_dir=d["data_dir"]
    weights=d["weights"]
    product=d["product"]
    overlapping=d["overlapping"]
    tile_size=d["tile_size"]
    features=d["features"]
    batch_size=d["batch_size"]




