# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# Copy geo-reference from an input file and apply it to the prediction output, using RasterIO.
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

import rasterio


def proj_rasterio(image, tif_mosaic):
    original_img = rasterio.open(image)

    # Extract spatial metadata
    input_crs = original_img.crs
    input_gt = original_img.transform

    # Read first band of input dataset
    processed_img = rasterio.open(tif_mosaic).read(1)

    # Prepare output geotiff file. We give crs and gt read from input as spatial metadata
    with rasterio.open(
            tif_mosaic,
            'w',
            driver='GTiff',
            count=1,
            height=processed_img.shape[0],
            width=processed_img.shape[1],
            dtype=processed_img.dtype,
            crs=input_crs,
            transform=input_gt
    ) as output:
        output.write(processed_img, 1)
