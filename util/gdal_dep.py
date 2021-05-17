import os


def get_projection (image_list, big_im_path, tif_mosaic):
    '''
    1) Open any .jp2 file from initial product (10m band), transform it into GeoTiff
    2) Get projection
    3) Apply it for the final prediction mosaic in .tif format
    '''
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