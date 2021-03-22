from pathlib import Path
import glob
import os
from PIL import Image
from math import ceil, floor
import re
import numpy as np
import pandas as pd
import pathlib

#create a list for images and take them in the right order

def get_img_entry_id(var):
    m = re.search(r'tile_(\d+)_(\d+)', str(var))
    if m:
        key = int(m.group(1)) * 1000 + int(m.group(2))
        return key
    return 0


# Create a function which rotates each image in the directory (if needed!!!)

def rotateImages(rotationAmt, img_list):
    # for each image in the current directory
    for image in img_list:
        # open the image
        img = Image.open(image)
        # rotate and save the image with the same filename
        img.rotate(rotationAmt).save(image)
        # close the image
        img.close()

#rotateImages(90) #In this case the images seem to be rotated 270˚ counter clockwise

#Rotate the final image
def rotate_img(img_path, rotate_degr):
    image = Image.open(img_path)
    image.rotate(rotate_degr, expand=1).save(img_path)
    image.close()


#Image grid

def image_grid(image_list, rows, cols):
    w, h = Image.open(image_list[0]).size

    new_img = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = new_img.size

    for i, img in enumerate(image_list):
        img = Image.open(img)

        new_img.paste(img, box=(i % cols * w, i // cols * h))
    return new_img


