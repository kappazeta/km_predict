import json
import argparse
from util import log as ulog
from architectures import ARCH_MAP
from data_generator import DataGenerator
from util.normalization import set_normalization
from util.save_prediction_masks import save_masks_contrast
import os
import numpy as np
from util.raster_mosaic import get_img_entry_id, rotateImages, rotate_img, image_grid, image_grid_overlap
from util.rasterio_dep import proj_rasterio
from util.gdal_dep import proj_gdal
import glob
import pathlib
from PIL import Image, ImageOps, ImageFile
from PIL.PngImagePlugin import PngImageFile, PngInfo
from math import ceil, floor
import subprocess
import shutil
import rasterio
from version import __version__



class CMPredict(ulog.Loggable):
    def __init__(self, log_abbrev="CMP.P"):
        super().__init__(log_abbrev)
        self.cfg = {
            "data_dir": ".SAFE",
            "product": "L2A",
            "overlapping": True,
            "tile_size": 512,
            "batch_size": 1
        }
        self.cm_vsm_executable = "cm_vsm"
        self.product_name = ""
        self.data_folder = "data"
        self.weigths_folder = "weights"
        self.predict_folder = "prediction"
        self.big_image_folder = "prediction"
        self.weights = ""
        self.product = "L2A"
        self.overlapping = True
        self.tile_size = 512
        self.resampling_method = "sinc"
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
        self.cm_vsm_version = 0

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
        if not os.path.exists(self.prediction_product_path):
            os.mkdir(self.prediction_product_path)

    def config_from_dict(self, d, product_name):
        """
        Load configuration from a dictionary.
        :param d: Dictionary with the configuration tree.
        """
        self.cm_vsm_executable = d["cm_vsm_executable"]
        if product_name:
            self.product_name = product_name
        else:
            self.product_name = d["product_name"]
        if d["level_product"] == "L2A":
            self.weights = "l2a__030-0.07.hdf5"
            self.features = ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B08", "B8A", "B09", "B11", "B12", "WVP"]
        elif d["level_product"] == "L1C":
            self.weights = "l1c__091-0.42.hdf5"
            self.features = ["B01", "B02", "B03", "B04", "B05", "B06", "B08", "B8A", "B09", "B10", "B11", "B12"]
        self.product = d["level_product"]
        self.overlapping = d["overlapping"]
        self.tile_size = d["tile_size"]
        self.resampling_method = d["resampling_method"]
        self.batch_size = d["batch_size"]
        self.architecture = d["architecture"]
        self.data_folder = d["folder_name"]

        self.product_safe = os.path.join(self.data_folder, str(self.product_name + ".SAFE"))
        self.weights_path = os.path.join(self.weigths_folder, self.weights)
        self.prediction_product_path = os.path.join(self.predict_folder, self.product_name)

        self.product_cvat = os.path.join(self.data_folder, (self.product_name + ".CVAT"))

    def load_config(self, path, product_name):
        with open(path, "rt") as fi:
            self.cfg = json.load(fi)
        self.config_from_dict(self.cfg, product_name)
        self.create_folders()

    def get_model_by_name(self, name):
        if self.architecture in ARCH_MAP:
            self.model = ARCH_MAP[name]()
            return self.model
        else:
            raise ValueError(("Unsupported architecture \"{}\"."
                              " Only the following architectures are supported: {}.").format(name, ARCH_MAP.keys()))

    def sub_tile(self, path_out):
        """
        Execute cm-vsm sub-tiling process
        """
        cm_vsm_query = (
            "{path_bin} -j -1 -d {path_in} -b {bands} -S {tile_size} -f 0 -m {resampling} -o {overlap}"
        ).format(
            path_bin=self.cm_vsm_executable,
            path_in=os.path.abspath(self.product_safe),
            bands=",".join(self.features),
            tile_size=self.tile_size,
            resampling=self.resampling_method,
            overlap=self.overlapping
        )
        if path_out and len(path_out) > 0:
            cm_vsm_query += " -O " + path_out
            self.product_cvat = path_out

        self.log.info("Performing CM-VSM:")
        with subprocess.Popen(cm_vsm_query, shell=True, stdout=subprocess.PIPE) as cm_vsm_process:
            for line in cm_vsm_process.stdout:
                cm_vsm_output = line.decode("utf-8").rstrip("\n")
                self.log.info(cm_vsm_output)
                if "Version:" in cm_vsm_output:
                    self.cm_vsm_version = cm_vsm_output.split(":")[1]
        self.log.info("Sub-tiling has been done!")

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

        # Look for .nc file, as the name is not specified
        for subfolder in os.listdir(self.product_cvat):
            subfolder_path = os.path.join(self.product_cvat, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith(".nc"):
                        tile_paths.append(os.path.join(subfolder_path, file))

        # Initialize data generator
        self.params = {'path_input': self.product_cvat,
                       'batch_size': self.batch_size,
                       'features': self.features,
                       'tile_size': self.tile_size,
                       'num_classes': len(self.classes),
                       'product_level': self.product,
                       'shuffle': False
                       }
        predict_generator = DataGenerator(tile_paths, **self.params)
        # sub_batch size 1 mean that we process data as whole, 2 dividing by half etc.
        #set_normalization(predict_generator, tile_paths, 1)
        # Run prediction
        predictions = self.model.predict(predict_generator)
        #sen2cor = predict_generator.get_sen2cor()
        #mask = (sen2cor[:, :, :, 3] == 1)
        #prediction_union = predictions
        #prediction_union[mask, 3] = sen2cor[mask, 3]
        y_pred = np.argmax(predictions, axis=3)
        for i, prediction in enumerate(predictions):
            save_masks_contrast(tile_paths[i], prediction, y_pred[i], self.prediction_product_path, self.classes)
        return

    def mosaic(self):
        """
        Make a mosaic output from obtained predictions
        Next step: Take into account overlapping argument
        """

        big_image_product = os.path.join(self.big_image_folder, self.product_name)
        if not os.path.exists(big_image_product):
            os.mkdir(big_image_product)

        # Create list of prediction images
        image_list = []

        for subfolder in os.listdir(self.prediction_product_path):
            if os.path.isdir(os.path.join(self.prediction_product_path, subfolder)):
                image_list.append(pathlib.Path(os.path.join(self.prediction_product_path, subfolder, "prediction.png")))

        # Sort images by asc
        image_list.sort(key = lambda var: get_img_entry_id(var))

        # Rotate each image in the list 270˚ counter clockwise
        rotateImages(270, image_list)

        # Raster mosaic
        """
        A function that creates raster mosaic.
        As parameters it takes: list of images, number of tiles per row and number of columns
        
        1) Takes the sub-tile width and height from the first image in the list
        2) Sets final image size from col*width, rows*height
        3) Creates final image from all sub-tiles, and bounding box parameters are also set 
        """
        new_im = image_grid_overlap(image_list, rows=23, cols=23, crop=16)

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = None

        jp2 = []
        if self.product == "L2A":
            for root, dirs, files in os.walk(self.product_safe):
                if(root.endswith("R10m")):
                    for file in files:
                        if(file.endswith(".jp2")):
                            jp2.append(os.path.join(root, file))
        elif self.product == "L1C":
            for root, dirs, files in os.walk(self.product_safe):
                if(root.endswith("IMG_DATA")):
                    for file in files:
                        if(file.endswith(".jp2")):
                            jp2.append(os.path.join(root, file))

        # Define a directory where to save a new file, resolution, etc.

        # Get name and index from product name
        date_name = self.product_name.rsplit('_', 4)[0].rsplit('_', 1)[1]
        index_name = self.product_name.rsplit('_', 1)[0].rsplit('_', 1)[-1]

        # Define the output names
        png_name = big_image_product + "/" + self.product + "_" + index_name + "_" + date_name + '_KZ_10m.png'
        tif_name = big_image_product + "/" + self.product + "_" + index_name + "_" + date_name + '_KZ_10m.tif'

        new_im.save(png_name, "PNG", quality=10980, optimize=True, progressive=True)
        new_im.save(tif_name, "TIFF", quality=10980, optimize=True, progressive=True)

        # Rotate final mosaic for 90˚ counter clockwise
        rotate_img(png_name, 90)
        rotate_img(tif_name, 90)

        png_mos = Image.open(png_name)
        tif_mos = Image.open(tif_name)

        # Flip final mosaic horisontally
        png_flip = ImageOps.flip(png_mos)
        tif_flip = ImageOps.flip(tif_mos)

        # Crop invalid pixels
        png_crop = ImageOps.crop(png_flip, (0, 0, 60, 60))
        tif_crop = ImageOps.crop(tif_flip, (0, 0, 60, 60))

        # Save final files
        png_crop.save(png_name)
        tif_crop.save(tif_name)

        proj_rasterio(jp2, tif_name)

        '''
        Assign 0-255 to 0-5 output
        Save final single band raster
        '''
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

            with rasterio.open(tif_name, 'w', **profile) as dst:
                dst.write(band1, 1)

        # Adding a version tag
        tif_img = Image.open(tif_name)
        tif_img.tag[305] = "CM_PREDICT V. {}; CM_VSM V. {}".format(__version__, str(self.cm_vsm_version).strip())
        tif_img.save(tif_name,tiffinfo=tif_img.tag)

        png_img = PngImageFile(png_name)
        metadata = PngInfo()
        metadata.add_text("Software", "CM_PREDICT V. {}; CM_VSM V. {}".format(__version__, str(self.cm_vsm_version).strip()))
        png_img.save(png_name, pnginfo=metadata)

        # Save 1 channel in final output

        # Create big_image/product_name folder with os.mkdir
        # Gather sub-tiles prediction from predict/product_name
        # Create image mosaic (preferably write in a separate file under /util)

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("-c", "--config", action="store", dest="path_config", help="Path to the configuration file.")
    p.add_argument("-product", "--product", action="store", dest="product_name",
                   help="Optional argument to overwrite product name in config.")
    p.add_argument("-t", "--no-tiling", action="store_true", dest="no_sub_tiling", default=False,
                   help="Disable sub-tiling (the tile output directory has already been created).")
    p.add_argument("-v", "--verbosity", action="store", dest="verbosity", default=1,
                   help="Verbosity level for logging: 0-WARNING, 1-INFO, 2-DEBUG. Default is 1.")
    p.add_argument("-l", "--log-file", action="store", dest="log_file_path", default=os.path.join(pathlib.Path(__file__).parent.absolute(), 'cm_predict.log'),
                   help="Optional argument to specify a location for .log file.")
    p.add_argument("-O", "--tiling-output", action="store", dest="path_out_tiling",
                   help="Override the path to the tiling output directory.")

    args = p.parse_args()
    ulog.init_logging(int(args.verbosity), "cm_predict", "CMP", args.log_file_path)
    cmf = CMPredict()
    cmf.load_config(args.path_config, args.product_name)
    if not args.no_sub_tiling:
        cmf.sub_tile(args.path_out_tiling)
    cmf.predict()
    cmf.mosaic()


if __name__ == "__main__":
    main()
