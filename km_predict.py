# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# KappaMask predictor.
#
# Copyright 2021 - 2022 KappaZeta Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import argparse
from util import log as ulog
from architectures import ARCH_MAP
from data_generator import DataGenerator
from util.normalization import set_normalization
from util.save_prediction_masks import save_masks_contrast
import os
import numpy as np
from util.raster_mosaic import get_img_entry_id, image_grid_overlap
from util.rasterio_dep import proj_rasterio
import pathlib
from PIL import Image, ImageOps, ImageFile
from PIL.PngImagePlugin import PngInfo
import subprocess
import rasterio
from version import __version__, min_cm_vsm_version
from pkg_resources import parse_version
import math
import tensorflow as tf
import urllib.request
import xml.etree.ElementTree as ET


class KMPredict(ulog.Loggable):
    def __init__(self, log_abbrev="KMP.P"):
        super().__init__(log_abbrev)
        self.cfg = {
            "data_dir": ".SAFE",
            "product": "L2A",
            "overlapping": 0.0625,
            "tile_size": 512,
            "batch_size": 1,
            "model_weights_source": "http://kappamask.s3-website.eu-central-1.amazonaws.com/model_weights/2022-06-16",
        }
        self.cm_vsm_executable = "cm_vsm"
        self.cm_vsm_env = None
        self.product_name = ""
        self.data_folder = "data"
        self.weights_folder = "weights"
        self.predict_folder = "prediction"
        self.big_image_folder = "prediction"
        self.weights = ""
        self.product = "L2A"
        self.overlapping = 0.0625
        self.tile_size = 512
        self.resampling_method = "sinc"
        self.features = [
            "AOT",
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
            "WVP",
        ]
        self.classes = [
            "UNDEFINED",
            "CLEAR",
            "CLOUD_SHADOW",
            "SEMI_TRANSPARENT_CLOUD",
            "CLOUD",
            "MISSING",
        ]
        self.batch_size = 1
        self.product_safe = ""
        self.product_cvat = ""
        self.weights_path = ""
        self.prediction_product_path = ""
        self.architecture = "Unet"
        self.params = {
            "path_input": self.product_cvat,
            "batch_size": self.batch_size,
            "features": self.features,
            "dim": self.tile_size,
            "num_classes": len(self.classes),
        }
        self.cm_vsm_version = "-"
        self.model = None
        self.aoi_geom = None
        self.model_weights_source = "http://kappamask.s3-website.eu-central-1.amazonaws.com/model_weights/2022-09-13"

    def create_folders(self):
        """
        Create data and weights folders if they do not exist
        """
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)
        if not os.path.exists(self.weights_folder):
            os.mkdir(self.weights_folder)
        if not os.path.exists(self.predict_folder):
            os.mkdir(self.predict_folder)
        if not os.path.exists(self.big_image_folder):
            os.mkdir(self.big_image_folder)
        if not os.path.exists(self.prediction_product_path):
            os.mkdir(self.prediction_product_path)

    def config_from_dict(self, d, product_name):
        """
        Load configuration from a dictionary.
        :param d: Dictionary with the configuration tree.
        :param product_name: Sentinel-2 product name.
        """
        if "cm_vsm_executable" in d:
            self.cm_vsm_executable = d["cm_vsm_executable"]
        elif "cm_vsm" in d:
            if "path" in d["cm_vsm"]:
                self.cm_vsm_executable = d["cm_vsm"]["path"]
            if "env" in d["cm_vsm"]:
                self.cm_vsm_env = d["cm_vsm"]["env"]

        if product_name:
            self.product_name = product_name
        else:
            self.product_name = d["product_name"]

        self.weights = "%s_%s.hdf5" % (
            d["level_product"].lower(),
            d["architecture"].lower(),
        )
        if d["level_product"] == "L2A":
            if d["architecture"] == "DeepLabv3Plus":
                self.features = [
                    "AOT",
                    "B01",
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B06",
                    "B07",
                    "B08",
                    "B8A",
                    "B09",
                    "B11",
                    "B12",
                    "WVP",
                ]
            elif d["architecture"] == "Unet":
                self.features = [
                    "AOT",
                    "B01",
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B06",
                    "B08",
                    "B8A",
                    "B09",
                    "B11",
                    "B12",
                    "WVP",
                ]
        elif d["level_product"] == "L1C":
            self.features = [
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B09",
                "B10",
                "B11",
                "B12",
            ]
        self.product = d["level_product"]
        self.overlapping = d["overlapping"]
        self.tile_size = d["tile_size"]
        self.resampling_method = d["resampling_method"]
        self.batch_size = d["batch_size"]
        self.architecture = d["architecture"]
        self.data_folder = d["folder_name"]

        self.product_safe = os.path.join(
            self.data_folder, str(self.product_name + ".SAFE")
        )
        self.product_metadata = os.path.join(
            self.data_folder,
            str(self.product_name + ".SAFE"),
            "MTD_MSI%s.xml" % d["level_product"],
        )

        self.product_baseline = self.get_product_baseline(self.product_metadata)
        self.offsets = self.get_offset_list(self.product_metadata, self.features)

        # Access data

        self.weights_path = os.path.join(self.weights_folder, self.weights)
        self.prediction_product_path = os.path.join(
            self.predict_folder, self.product_name
        )

        self.product_cvat = os.path.join(
            self.data_folder, (self.product_name + ".CVAT")
        )

        if "aoi_geometry" in d:
            self.aoi_geom = d["aoi_geometry"]

        if "model_weights_source" in d:
            self.model_weights_source = d["model_weights_source"]

    def load_config(self, path, product_name):
        with open(path, "rt") as fi:
            self.cfg = json.load(fi)
        self.config_from_dict(self.cfg, product_name)
        self.create_folders()

        overlap_pix = self.overlapping * self.tile_size
        if (overlap_pix % 2) != 0:
            raise Exception("Even number of pixels needed")

    def get_model_by_name(self, name):
        if self.architecture in ARCH_MAP:
            self.model = ARCH_MAP[name]()
            return self.model
        else:
            raise ValueError(
                (
                    'Unsupported architecture "{}".'
                    " Only the following architectures are supported: {}."
                ).format(name, ARCH_MAP.keys())
            )

    def get_cm_vsm_version(self):
        """
        Get the version of the cm-vsm utility.
        """
        q = [self.cm_vsm_executable, "--version"]
        with subprocess.Popen(
            q, stdout=subprocess.PIPE, env=self.cm_vsm_env
        ) as cm_vsm_process:
            for line in cm_vsm_process.stdout:
                cm_vsm_output = line.decode("utf-8").rstrip("\n")
                if "Version:" in cm_vsm_output:
                    self.cm_vsm_version = cm_vsm_output.split(":")[1]
        return self.cm_vsm_version

    def get_model_weights(self):
        if not os.path.exists(self.weights_path):
            self.log.info("Downloading model weights {} ...".format(self.weights))
            url = os.path.join(self.model_weights_source, self.weights)
            site = urllib.request.urlopen(url)
            urllib.request.urlretrieve(url, self.weights_path)

    def get_product_baseline(self, filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        baseline = root.findall(".//PROCESSING_BASELINE")[0].text
        return baseline

    def get_offset_list(self, filepath, features):
        tree = ET.parse(filepath)
        root = tree.getroot()

        offsets = []
        offset_list = root.find(".//Radiometric_Offset_List")

        if not offset_list:
            offsets = np.zeros(
                len(features),
            )
            return offsets

        else:
            for child in offset_list:
                value = int(child.text.strip()) if child.text else None
                if value is not None:
                    offsets.append(value)

            return np.array(offsets)

    def sub_tile(self, path_out, aoi_geom):
        """
        Execute cm-vsm sub-tiling process
        """
        if aoi_geom is not None:
            self.aoi_geom = aoi_geom

        cm_vsm_query = [
            self.cm_vsm_executable,
            "-j",
            "-1",
            "-d",
            os.path.abspath(self.product_safe),
            "-b",
            ",".join(self.features),
            "-S",
            str(self.tile_size),
            "-f",
            "0",
            "-m",
            self.resampling_method,
            "-o",
            str(self.overlapping),
        ]
        if path_out and len(path_out) > 0:
            cm_vsm_query += ["-O", path_out]
            self.product_cvat = path_out
        # Area of interest geometry supplied?
        if self.aoi_geom is not None:
            cm_vsm_query += ["-g", self.aoi_geom]

        self.log.info("Splitting with CM-VSM: {}".format(cm_vsm_query))
        self.log.info("Product processing baseline: %s" % self.product_baseline)
        with subprocess.Popen(
            cm_vsm_query, stdout=subprocess.PIPE, env=self.cm_vsm_env
        ) as cm_vsm_process:
            for line in cm_vsm_process.stdout:
                cm_vsm_output = line.decode("utf-8").rstrip("\n")
                self.log.info(cm_vsm_output)
        self.log.info("Sub-tiling has been done!")

    def predict(self, force_predict=False):
        """
        Run prediction for every sub-folder
        """
        # Initialize model
        self.get_model_by_name(self.architecture)

        # Propagate configuration parameters.
        self.model.set_batch_size(self.batch_size)

        # Construct and compile the model.
        self.model.construct(
            self.tile_size, self.tile_size, len(self.features), len(self.classes)
        )
        self.model.compile()

        # Load model weights.
        self.model.load_weights(self.weights_path)

        # Go through all folders
        date_match = self.product_name.rsplit("_", 1)[-1]
        index_match = self.product_name.rsplit("_", 1)[0].rsplit("_", 1)[-1]

        tile_paths = []

        # Look for .nc file, as the name is not specified
        for subfolder in os.listdir(self.product_cvat):
            subfolder_path = os.path.join(self.product_cvat, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith(".nc"):
                        tile_paths.append(os.path.join(subfolder_path, file))

        # Initialize data generator
        self.params = {
            "path_input": self.product_cvat,
            "architecture": self.architecture,
            "batch_size": self.batch_size,
            "features": self.features,
            "tile_size": self.tile_size,
            "num_classes": len(self.classes),
            "product_level": self.product,
            "offsets": self.offsets,
            "shuffle": False,
        }

        # Check if prediction already exists
        tile_paths_unseen = []
        for tp in tile_paths:
            path_image = tp.split("/")[-2:-1][0]
            prediction_path = os.path.join(
                self.prediction_product_path, path_image, "prediction.png"
            )

            # If True, run prediction on all tiles, else only on those for which prediction.png does not exist
            if force_predict:
                tile_paths_unseen.append(tp)
            else:
                if not os.path.exists(prediction_path):
                    tile_paths_unseen.append(tp)

        # Predict in batches
        for j in range(0, len(tile_paths_unseen), self.batch_size):
            tile_paths_subset = tile_paths_unseen[j : (j + self.batch_size)]
            self.params["batch_size"] = len(tile_paths_subset)
            predict_generator = DataGenerator(tile_paths_subset, **self.params)

            # Run prediction
            predictions = self.model.predict(predict_generator)
            y_pred = np.argmax(predictions, axis=3)
            for i, prediction in enumerate(predictions):
                save_masks_contrast(
                    tile_paths_subset[i],
                    prediction,
                    y_pred[i],
                    self.prediction_product_path,
                    self.classes,
                )
        return

    def mosaic(self):
        """
        Make a mosaic output from obtained predictions with an overlapping argument
        """
        # Create /prediction/<product_name> directory
        big_image_product = os.path.join(self.big_image_folder, self.product_name)
        if not os.path.exists(big_image_product):
            os.mkdir(big_image_product)

        # Create list of prediction images
        image_list = []
        for subfolder in os.listdir(self.prediction_product_path):
            if os.path.isdir(os.path.join(self.prediction_product_path, subfolder)):
                image_list.append(
                    pathlib.Path(
                        os.path.join(
                            self.prediction_product_path, subfolder, "prediction.png"
                        )
                    )
                )

        # Sort images by asc (e.g. 0_0, 0_1, 0_2)
        image_list.sort(key=lambda var: get_img_entry_id(var))

        """
        A function that creates raster mosaic.
        As parameters it takes: list of images, number of tiles per row and number of columns
        
        1) Takes the sub-tile width and height from the first image in the list
        2) Sets final image size from col*width, rows*height
        3) Creates final image from all sub-tiles, bounding box parameters are also set 
        """
        overlap_pix = self.overlapping * self.tile_size
        crop_coef = int(overlap_pix / 2)
        n_rows = math.ceil(10980 / (self.tile_size - crop_coef))
        new_im = image_grid_overlap(
            image_list, rows=n_rows, cols=n_rows, crop=crop_coef
        )

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = None

        # For a correct georeference it is necessary to use 10m resolution band
        jp2 = ""
        if self.product == "L2A":
            for root, dirs, files in os.walk(self.product_safe):
                if root.endswith("R10m"):
                    for file in files:
                        if file.endswith(".jp2"):
                            jp2 = os.path.join(root, file)
        elif self.product == "L1C":
            for root, dirs, files in os.walk(self.product_safe):
                if root.endswith("IMG_DATA"):
                    for file in files:
                        if file.endswith("B02.jp2"):
                            jp2 = os.path.join(root, file)

        # Define a directory where to save a new file, resolution, etc.
        # Get name and index from product name
        date_name = self.product_name.rsplit("_", 4)[0].rsplit("_", 1)[1]
        index_name = self.product_name.rsplit("_", 1)[0].rsplit("_", 1)[-1]

        # Define the output names
        png_name = os.path.join(
            big_image_product,
            self.product + "_" + index_name + "_" + date_name + "_KZ_10m.png",
        )
        tif_name = os.path.join(
            big_image_product,
            self.product + "_" + index_name + "_" + date_name + "_KZ_10m.tif",
        )

        # Crop the edges in the final image
        f_tile_size = (self.tile_size - crop_coef * 2) * n_rows
        crop = f_tile_size - 10980
        new_im_cropped = ImageOps.crop(new_im, (0, 0, crop, crop))

        # Fill metadata for PNG format
        metadata = PngInfo()
        metadata.add_text(
            "Software",
            "KM_PREDICT {}; CM_VSM {}".format(
                __version__, str(self.cm_vsm_version).strip()
            ),
        )

        # Save with a recommended quality and metadata for png, tif is done further down
        new_im_cropped.save(png_name, "PNG", quality=95, pnginfo=metadata)
        new_im_cropped.save(tif_name, "TIFF", quality=95)

        # Deal with tiff-related issues: projection, bands, tags
        proj_rasterio(jp2, tif_name)
        """
        Assign 0-255 to 0-5 output
        Save final single band raster
        """

        # Read band 1 (out of 3, they're identical)
        with rasterio.open(tif_name) as tif:
            profile = tif.profile.copy()
            band1 = tif.read(1)

            # Translate values
            band1[band1 == 0] = 0
            band1[band1 == 66] = 1
            band1[band1 == 129] = 2
            band1[band1 == 192] = 3
            band1[band1 == 255] = 4
            band1[band1 == 20] = 5

            profile.update({"count": 1})

            with rasterio.open(tif_name, "w", **profile) as dst:
                dst.write(band1, 1)

        # Add a version tag for tiff image
        tif_img = Image.open(tif_name)
        tif_img.tag[305] = "KM_PREDICT {}; CM_VSM {}".format(
            __version__, str(self.cm_vsm_version).strip()
        )
        tif_img.save(tif_name, tiffinfo=tif_img.tag)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "-c",
        "--config",
        action="store",
        dest="path_config",
        help="Path to the configuration file.",
    )
    p.add_argument(
        "-product",
        "--product",
        action="store",
        dest="product_name",
        help="Optional argument to override product name in config.",
    )
    p.add_argument(
        "-t",
        "--no-tiling",
        action="store_true",
        dest="no_sub_tiling",
        default=False,
        help="Disable sub-tiling (the tile output directory has already been created).",
    )
    p.add_argument(
        "-cpu",
        "--use-cpu",
        action="store_true",
        dest="use_cpu",
        default=False,
        help="Use CPU.",
    )
    p.add_argument(
        "-f",
        "--force-predict",
        action="store_true",
        dest="force_predict",
        default=False,
        help="Force prediction on the tiles for which prediction.png already exists.",
    )
    p.add_argument(
        "-v",
        "--verbosity",
        action="store",
        dest="verbosity",
        default=1,
        help="Verbosity level for logging: 0-WARNING, 1-INFO, 2-DEBUG. Default is 1.",
    )
    p.add_argument(
        "-l",
        "--log-file",
        action="store",
        dest="log_file_path",
        default=os.path.join(
            pathlib.Path(__file__).parent.absolute(), "km_predict.log"
        ),
        help="Optional argument to specify a location for .log file.",
    )
    p.add_argument(
        "-O",
        "--tiling-output",
        action="store",
        dest="path_out_tiling",
        help="Override the path to the tiling output directory.",
    )
    p.add_argument(
        "-g",
        "--geom",
        action="store",
        dest="aoi_geom",
        help='Area of interest geometry as an EWKT string, for subtiling. For example: "SRID=4326;Polygon '
        "((22.64992375534184887 50.27513740160615185, 23.60228115218003708 50.35482161490517683, "
        "23.54514084707420452 49.94024031630130622, 23.3153953947536472 50.21771699530808775, "
        '22.64992375534184887 50.27513740160615185))"',
    )

    args = p.parse_args()

    log = ulog.init_logging(
        int(args.verbosity), "km_predict", "KMP", args.log_file_path
    )

    if args.use_cpu:
        tf.config.set_visible_devices([], "GPU")

    if args.path_config is None:
        p.print_help()
        log.error("Expecting the path to a configuration file")
    else:
        kmf = KMPredict()
        kmf.load_config(args.path_config, args.product_name)
        cm_vsm_version = kmf.get_cm_vsm_version()

        # Ensure that we have a compatible version of cm-vsm.
        if parse_version(cm_vsm_version) < parse_version(min_cm_vsm_version):
            log.error("Please update cm-vsm to " + min_cm_vsm_version + " or later")
        else:
            kmf.get_model_weights()

            if not args.no_sub_tiling:
                kmf.sub_tile(args.path_out_tiling, args.aoi_geom)

            kmf.predict(force_predict=args.force_predict)
            kmf.mosaic()


if __name__ == "__main__":
    main()
