import json
import argparse
from util import log as ulog

class CMPredict(ulog.Loggable):
    def __init__(self):
        self.cfg={
          "data_dir": ".SAFE",
          "weights": "",
          "product": "L2A",
          "overlapping": True,
          "tile_size": 512,
          "features": ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B08", "B8A", "B09", "B11", "B12", "WVP"],
          "batch_size": 1
        }
        self.data_dir=".SAFE"
        self.weights=""
        self.product="L2A"
        self.overlapping=True
        self.tile_size=512
        self.features=["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B08", "B8A", "B09", "B11", "B12", "WVP"]
        self.batch_size=1

    def config_from_dict(self, d):
        """
        Load configuration from a dictionary.
        :param d: Dictionary with the configuration tree.
        """
        self.data_dir=d["data_dir"]
        self.weights=d["weights"]
        self.product=d["product"]
        self.overlapping=d["overlapping"]
        self.tile_size=d["tile_size"]
        self.features=d["features"]
        self.batch_size=d["batch_size"]

    def load_config(self,path):
        with open(path, "rt") as fi:
            self.cfg = json.load(fi)
        self.config_from_dict(self.cfg)

def main():
    p=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("-c", "--config", action="store", dest="path_config",help="Path to the configuration file.")
    args=p.parse_args()
    cmf = CMPredict()
    cmf.load_config(args.path_config)

if __name__ == "__main__":
    main()







