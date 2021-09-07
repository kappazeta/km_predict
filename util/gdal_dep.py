# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# Copy geo-reference from an input file and apply it to the prediction output, using GDAL.
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

import os


def proj_gdal(image_list, big_im_path, tif_mosaic):
    """
    1) Open any .jp2 file from initial product (10m band), transform it into GeoTiff
    2) Get projection
    3) Apply it for the final prediction mosaic in .tif format
    """
    import gdal
    in_image = gdal.Open(image_list[0])
    driver = gdal.GetDriverByName("GTiff")
    out_image = driver.CreateCopy((big_im_path + "/" + 'projection.tif'), in_image, 0)

    in_image = None
    out_image = None

    tif = gdal.Open(big_im_path + "/" + 'projection.tif')

    prj = tif.GetProjection()
    gt = tif.GetGeoTransform()

    mosaic = gdal.OpenShared(tif_mosaic, gdal.GA_Update)
    mosaic.SetProjection(prj)
    mosaic.SetGeoTransform(gt)

    # Delete the Geotiff projection/transformation file
    os.remove(big_im_path + "/" + 'projection.tif')

    return mosaic
