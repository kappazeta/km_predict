import json
import argparse
from util import log as ulog
from architectures import ARCH_MAP
from data_generator import DataGenerator
from util.normalization import set_normalization
from util.save_prediction_masks import save_masks_contrast
import os
import numpy as np


class CMPredict(ulog.Loggable):
    def __init__(self, log_abbrev="CMPred"):
        super().__init__(log_abbrev)
        self.cfg = {
            "data_dir": ".SAFE",
            "weights": "",
            "product": "L2A",
            "overlapping": True,
            "tile_size": 512,
            "features": ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B08", "B8A", "B09", "B11", "B12", "WVP"],
            "batch_size": 1
        }
        self.product_name = ""
        self.data_folder = "data"
        self.weigths_folder = "weights"
        self.predict_folder = "prediction"
        self.big_image_folder = "big_image"
        self.weights = ""
        self.product = "L2A"
        self.overlapping = True
        self.tile_size = 512
        self.features = ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B08", "B8A", "B09", "B11", "B12", "WVP"]
        self.classes = [
            "UNDEFINED", "CLEAR", "CLOUD_SHADOW", "SEMI_TRANSPARENT_CLOUD", "CLOUD", "MISSING"
        ]
        self.batch_size = 1
        self.product_safe = ""
        self.product_cvat = ""
        self.weights_path = ""
        self.prediction_product_path = ""
        self.architecture = "Unet"
        self.params = {'path_input': self.product_cvat,
                       'batch_size': self.batch_size,
                       'features': self.features,
                       'dim': self.tile_size,
                       'num_classes': len(self.classes)
                       }
        self.create_folders()

    def create_folders(self):
        """
        Create data and weights folders if they do not exist
        """
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)
        if not os.path.exists(self.weigths_folder):
            os.mkdir(self.weigths_folder)
        if not os.path.exists(self.predict_folder):
            os.mkdir(self.predict_folder)
        if not os.path.exists(self.big_image_folder):
            os.mkdir(self.big_image_folder)

    def config_from_dict(self, d):
        """
        Load configuration from a dictionary.
        :param d: Dictionary with the configuration tree.
        """
        self.product_name = d["product_name"]
        self.weights = d["weights"]
        self.product = d["product"]
        self.overlapping = d["overlapping"]
        self.tile_size = d["tile_size"]
        self.features = d["features"]
        self.batch_size = d["batch_size"]
        self.architecture = d["architecture"]

        self.product_safe = os.path.join(self.data_folder, str(self.product_name + ".SAFE"))
        self.weights_path = os.path.join(self.weigths_folder, self.weights)
        self.prediction_product_path = os.path.join(self.predict_folder, self.product_name)
        if not os.path.exists(self.prediction_product_path):
            os.mkdir(self.prediction_product_path)

    def load_config(self, path):
        with open(path, "rt") as fi:
            self.cfg = json.load(fi)
        self.config_from_dict(self.cfg)

    def get_model_by_name(self, name):
        if self.architecture in ARCH_MAP:
            self.model = ARCH_MAP[name]()
            return self.model
        else:
            raise ValueError(("Unsupported architecture \"{}\"."
                              " Only the following architectures are supported: {}.").format(name, ARCH_MAP.keys()))

    def sub_tile(self):
        """
        Execute cm-vsm sub-tiling process
        """
        self.product_cvat = os.path.join(self.data_folder, (self.product_name + ".CVAT"))

    def predict(self):
        """
        Run prediction for every sub-folder
        """
        # Initialize model

        self.get_model_by_name(self.architecture)

        # Propagate configuration parameters.
        self.model.set_batch_size(self.batch_size)

        # Construct and compile the model.
        self.model.construct(self.tile_size, self.tile_size, len(self.features), len(self.classes))
        self.model.compile()

        # Load model weights.
        self.model.load_weights(self.weights_path)

        # Go through all folders
        date_match = self.product_name.rsplit('_', 1)[-1]
        index_match = self.product_name.rsplit('_', 1)[0].rsplit('_', 1)[-1]

        tile_paths = []

        for subfolder in os.listdir(self.product_cvat):
            tile_paths.append(os.path.join(self.product_cvat, subfolder, "bands.nc"))

        # Initialize data generator
        self.params = {'path_input': self.product_cvat,
                       'batch_size': self.batch_size,
                       'features': self.features,
                       'tile_size': self.tile_size,
                       'num_classes': len(self.classes)
                       }
        predict_generator = DataGenerator(tile_paths, **self.params)
        # sub_batch size 1 mean that we process data as whole, 2 dividing by half etc.
        #set_normalization(predict_generator, tile_paths, 1)
        # Run prediction
        predictions = self.model.predict(predict_generator)
        y_pred = np.argmax(predictions, axis=3)
        for i, prediction in enumerate(predictions):
            save_masks_contrast(tile_paths[i], prediction, y_pred[i], self.prediction_product_path, self.classes)
        return

    def mosaic(self):
        """
        Make a mosaic output from obtained predictions
        Next step: Take into account overlapping argument
        """
        # Create big_image/product_name folder with os.mkdir
        # Gather sub-tiles prediction from predict/product_name
        # Create image mosaic (preferably write in a separate file under /util)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("-c", "--config", action="store", dest="path_config", help="Path to the configuration file.")
    args = p.parse_args()
    cmf = CMPredict()
    cmf.load_config(args.path_config)
    cmf.sub_tile()
    cmf.predict()
    cmf.mosaic()


if __name__ == "__main__":
    main()
