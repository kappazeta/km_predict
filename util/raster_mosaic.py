# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# KappaMask predictor version and changelog.
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

from PIL import Image, ImageOps
import re


def extract_tile_coords(image_path):
    m = re.search(r"tile_(\d+)_(\d+)", str(image_path))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


# Create a list for images and take them in the right order
def get_img_entry_id(var):
    x, y = extract_tile_coords(var)
    if x is not None and y is not None:
        return x * 1000 + y
    return 0


def image_grid(image_list, rows, cols):
    w, h = Image.open(image_list[0]).size

    new_img = Image.new("RGB", size=(cols * w, rows * h))

    for img in image_list:
        # Get row and column number
        col_id, row_id = extract_tile_coords(img)

        img = Image.open(img)

        new_img.paste(img, box=(row_id * w, col_id * h))
    return new_img


def image_grid_overlap(image_list, rows, cols, crop):
    w, h = Image.open(image_list[0]).size
    new_img = Image.new("RGB", size=(cols * (w - crop * 2), rows * (h - crop * 2)))

    # Creates new empty image with taking the size from sub-tile from the list
    for img in image_list:
        # Get row and column number
        col_id, row_id = extract_tile_coords(img)
        if col_id is None or row_id is None:
            continue

        img = Image.open(img)

        border = (crop, crop, crop, crop)  # left, up, right, bottom

        # Set conditions to crop images differently.
        # Corners should be cropped first, then the borders
        if row_id == 0 and col_id == 0:
            img_crop = ImageOps.crop(img, (0, 0, crop, crop))  # upper left corner

        elif col_id == 0 and row_id == (rows - 1):
            img_crop = ImageOps.crop(img, (0, crop, crop, 0))  # lower left corner

        elif row_id == 0 and col_id == (cols - 1):
            img_crop = ImageOps.crop(img, (crop, 0, 0, crop))  # upper right corner

        elif row_id == (rows - 1) and col_id == (cols - 1):
            img_crop = ImageOps.crop(img, (crop, crop, 0, 0))  # lower right corner

        elif row_id == 0:
            img_crop = ImageOps.crop(img, (crop, 0, crop, crop))  # 1st row

        elif col_id == 0:
            img_crop = ImageOps.crop(img, (0, crop, crop, crop))  # 1st column

        elif row_id == (rows - 1):
            img_crop = ImageOps.crop(img, (crop, crop, crop, 0))  # last row

        elif col_id == (cols - 1):
            img_crop = ImageOps.crop(img, (crop, crop, 0, crop))  # last column

        else:
            img_crop = ImageOps.crop(img, border)

        # When all images in the list cropped, paste them in the new empty image
        """
        For the first tile set the offset 0,0
        As most of tiles are cropped 16 pixels from all 4 sides the offset should be shifted crop*2 on X and Y axis
        Border tiles are cropped differently, so the offsets are shifted only for "crop" argument
        """
        if col_id == 0:
            off_x = 0
        elif col_id == 1:
            off_x = w - crop
        else:
            off_x = (w - crop) + (col_id - 1) * (w - crop * 2)

        if row_id == 0:
            off_y = 0
        elif row_id == 1:
            off_y = h - crop
        else:
            off_y = (h - crop) + (row_id - 1) * (h - crop * 2)

        box = (off_x, off_y)

        # Image is built column by column, vertically, starting from (0.0) left-upper corner
        new_img.paste(img_crop, box=box)
    return new_img
