# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# Save cloud mask raster.
#
# Copyright 2021 KappaZeta Ltd.
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

from PIL import Image
import os
import skimage.io as skio
import numpy as np


def save_masks_contrast(path_image, prediction, classification, saving_path, classes):
    path_image = path_image.rstrip()
    filename_image = path_image.split('/')[-2:-1][0]
    if "tile" in (saving_path + "/" + filename_image):
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)
        if not os.path.exists(saving_path + "/" + filename_image):
            os.mkdir(saving_path + "/" + filename_image)
        for i, label in enumerate(classes):
            saving_filename = saving_path + "/" + filename_image + "/predict_" + label
            current_class = prediction[:, :, i]
            current_class *= 255
            current_class = current_class.astype(np.uint8)
            current_class = np.flip(current_class, 0)
            current_class = Image.fromarray(current_class)
            current_class.save(saving_filename + ".png", compress_level = 4)
        classification = classification *63 + 3
        classification[classification > 255] = 20
        classification = classification.astype(np.uint8)
        im = Image.fromarray(classification)
        im.save(saving_path + "/" + filename_image + "/prediction.png")
    return True
