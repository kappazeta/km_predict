import rasterio

def proj_rasterio (image_list, tif_mosaic):
    original_img = rasterio.open(image_list[0])

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