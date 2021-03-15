import json
import argparse

p=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

p.add_argument("-c", "--config", action="store", dest="path_config",help="Path to the configuration file.")

args=p.parse_args()

with open(args.path_config, "rt") as fi:
    cfg = json.load(fi)

print(cfg["features"])


def load_config(self,path):
    """
    Load configuration from a JSON file.
    :param path: Path to the JSON file.
    """
    with open(path, "rt") as fi:
        self.cfg = json.load(fi)
        # TODO:: Validate config structure
        self.config_from_dict(self.cfg)

