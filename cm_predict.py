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
from math import ceil, floor
import subprocess
import shutil
import rasterio



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
        self.big_image_folder = "full_mosaic"
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
        if product_name:
            self.product_name = product_name
        else:
            self.product_name = d["product_name"]
        self.weights = d["weights"]
        self.product = d["level_product"]
        self.overlapping = d["overlapping"]
        self.tile_size = d["tile_size"]
        self.features = d["features"]
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

    def sub_tile(self):
        """
        Execute cm-vsm sub-tiling process
        """
        cm_vsm_query = \
            self.cfg["cm_vsm_executable"] + \
            " -j -1 " + \
            " -d " + os.path.abspath(self.product_safe) + \
            " -b " + ",".join(self.cfg["features"]) + \
            " -S " + str(self.cfg["tile_size"]) + \
            " -f 0" + \
            " -m " + self.cfg["resampling_method"] + \
            " -o " + str(self.cfg["overlapping"])
        temp_logs_path = self.data_folder + "/" + self.product_name + ".log"
        final_logs_path = self.product_cvat + "/" + self.product_name + ".log"

        print("Starting cm-vsm...")
        with subprocess.Popen(cm_vsm_query, shell=True, stdout=subprocess.PIPE) as cm_vsm_process:
            for line in cm_vsm_process.stdout:
                with open(temp_logs_path, 'a') as file:
                    file.write(line.decode("utf-8"))
        shutil.move(temp_logs_path, final_logs_path)
        print("Sub-tiling has been done. Log file is avaliable in the .CVAT folder.", )

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
                       'shuffle': False
                       }
        predict_generator = DataGenerator(tile_paths, **self.params)
        # sub_batch size 1 mean that we process data as whole, 2 dividing by half etc.
        #set_normalization(predict_generator, tile_paths, 1)
        # Run prediction
        predictions = self.model.predict(predict_generator)
        sen2cor = predict_generator.get_sen2cor()
        mask = (sen2cor[:, :, :, 3] == 1)
        prediction_union = predictions
        prediction_union[mask, 3] = sen2cor[mask, 3]
        y_pred = np.argmax(prediction_union, axis=3)
        for i, prediction in enumerate(prediction_union):
            save_masks_contrast(tile_paths[i], prediction, y_pred[i], self.prediction_product_path, self.classes)
        return

    def mosaic(self):
        """
        Make a mosaic output from obtained predictions
        Next step: Take into account overlapping argument
        """
        #self.product_name #tile name
        # self.big_image_folder#big image (directory)

        self.big_image_product = self.big_image_folder + "/" + self.product_name
        if not os.path.exists(self.big_image_product):
            os.mkdir(self.big_image_product)

        # self.predict_folder #working_path

        file_names =['prediction']

        # Create list of prediction images
        image_list = []
        tile_paths = []

        for subfolder in os.listdir(self.prediction_product_path):
            image_list.append(pathlib.Path(os.path.join(self.prediction_product_path, subfolder, "prediction.png")))
        #image_list = [y for x in file_names for y in pathlib.Path(self.predict_folder).glob(f'**/{x}*.png')]

        image_list.sort(key=lambda var: get_img_entry_id(var))

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

        #1
        jp2 = []

        for root, dirs, files in os.walk(self.product_safe):
            if(root.endswith("R10m")):
                for file in files:
                    if(file.endswith(".jp2")):
                        jp2.append(os.path.join(root, file))

        # Define a directory where to save a new file, resolution, etc.

        #Get name and index from product name
        date_name = self.product_name.rsplit('_', 4)[0].rsplit('_', 1)[1]
        index_name = self.product_name.rsplit('_', 1)[0].rsplit('_', 1)[-1]

        #Define the output names
        png_name = self.big_image_product +"/" +"L2A_"+index_name+"_"+date_name +'_KZ_10m.png'
        tif_name = self.big_image_product +"/" +"L2A_"+index_name+"_"+date_name +'_KZ_10m.tif'

        new_im.save(png_name, "PNG", quality=10980, optimize=True, progressive=True)
        new_im.save(tif_name, "TIFF", quality=10980, optimize=True, progressive=True)

        # Rotate final mosaic for 90˚ counter clockwise
        rotate_img(png_name, 90)
        rotate_img(tif_name, 90)

        png_mos = Image.open(png_name)
        tif_mos = Image.open(tif_name)


        #Flip final mosaic horisontally
        png_flip = ImageOps.flip(png_mos)
        tif_flip = ImageOps.flip(tif_mos)

        #Crop invalid pixels
        png_crop = ImageOps.crop(png_flip, (0, 0, 60, 60))
        tif_crop = ImageOps.crop(tif_flip, (0, 0, 60, 60))


        #Save final files
        png_crop.save(png_name)
        tif_crop.save(tif_name)

        proj_rasterio(jp2, tif_name)
        #proj_gdal(jp2, self.big_image_folder, tif_name)

        '''Assign 0-255 to 0-5 output
           Save final single band raster'''

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
                   help="Disable sub-tiling if CVAT folder is already created.")

    args = p.parse_args()
    cmf = CMPredict()
    cmf.load_config(args.path_config, args.product_name)
    if not args.no_sub_tiling:
        cmf.sub_tile()
    cmf.predict()
    cmf.mosaic()


if __name__ == "__main__":
    main()
