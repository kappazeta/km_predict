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
            # current_class[current_class >= 0.5] = 255
            # current_class[current_class < 0.5] = 0
            current_class *= 255
            current_class = current_class.astype(np.uint8)
            skio.imsave(saving_filename +".png", current_class)
        classification = classification *63 + 3
        classification[classification > 255] = 20
        classification = classification.astype(np.uint8)
        # skio.imsave(saving_path + "/" + filename_image + "/prediction.png", classification)
        im = Image.fromarray(classification)
        im.save(saving_path + "/" + filename_image + "/prediction.png")
    return True