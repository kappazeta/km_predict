from pathlib import Path
import glob
import os
from PIL import Image, ImageOps
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


def image_grid_overlap(image_list, rows, cols, crop):
    w, h = Image.open(image_list[0]).size
    new_img = Image.new('RGB', size=(cols * (w - crop * 2), rows * (h - crop * 2)))
    grid_w, grid_h = new_img.size

    off_x = 0
    off_y = 0

    # Creates new empty image with taking the size from sub-tile from the list

    for i, img in enumerate(image_list):
        img = Image.open(img)

        border = (crop, crop, crop, crop)  # left, up, right, bottom

        # Get row and column number
        col_id = i % cols
        row_id = i // cols

        # Set conditions to crop images differently.
        # Corners should be cropped first, then the borders

        if (row_id == 0 and col_id == 0):
            img_crop = ImageOps.crop(img, (0, 0, crop, crop))  # upper left corner

        elif (i % cols == 0 and row_id == (rows - 1)):
            img_crop = ImageOps.crop(img, (0, crop, crop, 0))  # lower left corner

        elif (row_id == 0 and col_id == (cols - 1)):
            img_crop = ImageOps.crop(img, (crop, 0, 0, crop))  # upper right corner

        elif (row_id == (rows - 1) and col_id == (cols - 1)):
            img_crop = ImageOps.crop(img, (crop, crop, 0, 0))  # lower right corner

        elif row_id == 0:
            img_crop = ImageOps.crop(img, (crop, 0, crop, crop))  # 1st row

        elif (col_id == 0):
            img_crop = ImageOps.crop(img, (0, crop, crop, crop))  # 1st column

        elif row_id == (rows - 1):
            img_crop = ImageOps.crop(img, (crop, crop, crop, 0))  # last row

        elif col_id == (cols - 1):
            img_crop = ImageOps.crop(img, (crop, crop, 0, crop))  # last column

        else:
            img_crop = ImageOps.crop(img, border)

        # When all images in the list cropped, paste them in the new empty image

        box = (off_x, off_y)
        new_img.paste(img_crop, box=box)

        '''
        For the first tile set the offset 0,0
        As the tiles are pasted row by row, every time it will go to first row x offset is reset to 0

        As most of tiles are cropped 16 pixels from all 4 sides the offset should be shifted crop*2 on X and Y axis
        Border tiles are cropped differently, so the offsets are shifted only for "crop" argument
        '''
        if col_id == 0:
            off_x += w - crop
        elif col_id == cols - 1:
            off_x = 0
        else:
            off_x += w - crop * 2

        if col_id == cols - 1:
            if row_id == 0:
                off_y += h - crop
            else:
                off_y += h - crop * 2

    return new_img

