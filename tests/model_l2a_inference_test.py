import pytest
import test
import numpy as np
from pathlib import Path
import urllib.request
from km_predict.architectures import DeepLabv3Plus
import tensorflow as tf
import netCDF4 as nc
from PIL import Image
from sklearn.metrics import f1_score


def deeplab_instance():
    # Create an instance of DeepLabv3Plus for testing
    return DeepLabv3Plus()


def download_weights_from_s3(weights_name, max_retries=5):
    # Download weights
    model_weights_url = f"http://kappamask.s3-website.eu-central-1.amazonaws.com/model_weights/2022-09-13/{weights_name}"

    # Folder to download weights to
    weights_path = Path("weights")
    weights_path.mkdir(exist_ok=True)

    if (weights_path / weights_name).exists():
        return
    else:
        retries = 0
        while retries < max_retries:
            try:
                site = urllib.request.urlopen(model_weights_url)
                urllib.request.urlretrieve(
                    model_weights_url, weights_path / weights_name
                )
                return
            except:
                retries += 1


def download_test_data_from_s3(test_filename, max_retries=5):

    # Output dir
    test_data_path = Path("test_l2a_data")
    test_data_path.mkdir(exist_ok=True)

    subtile_path_url = f"http://kappamask.s3-website.eu-central-1.amazonaws.com/test_data/test_inference_l2a/{test_filename}"

    if (test_data_path / test_filename).exists():
        return
    else:
        retries = 0
        while retries < max_retries:
            try:
                site = urllib.request.urlopen(subtile_path_url)
                urllib.request.urlretrieve(
                    subtile_path_url, test_data_path / test_filename
                )
                return
            except:
                retries += 1


def load_and_normalize(img_filepath):
    min_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_v = [
        0.010711833834648132,
        0.32552072405815125,
        0.3022507131099701,
        0.28221559524536133,
        0.2690470814704895,
        0.27560845017433167,
        0.2571450471878052,
        0.2557412087917328,
        0.2529335618019104,
        0.2544899582862854,
        0.2657511234283447,
        0.2381628155708313,
        0.2401922643184662,
        0.14035248756408691,
    ]
    features = [
        "AOT",
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
        "WVP",
    ]

    with nc.Dataset(img_filepath, "r") as root:
        data_bands = [
            (np.asarray(root[f]) - min_v[i]) / (max_v[i] - min_v[i])
            for i, f in enumerate(features)
        ]
        label = np.asarray(root["KML2A"])
    data_bands = np.stack(data_bands)
    data_bands = np.rollaxis(data_bands, 0, 3)

    return data_bands, label


def test_inference():

    assert hasattr(
        deeplab_instance(), "construct"
    ), "DeepLabv3Plus class should have a 'construct' method"

    weights_name = "l2a_deeplabv3plus.hdf5"
    subtiles_names = [
        "T20MMT_20200505T142729_tile_10_12.nc",
        "T12SUG_20201009T181251_tile_7_1.nc",
        "T22KCF_20200619T134219_tile_1_8.nc",
        "T17SNA_20200312T160949_tile_4_9.nc",
        "T30TWL_20200702T105621_tile_4_21.nc",
    ]

    # Download weights
    download_weights_from_s3(weights_name)

    # Load model
    width = 512
    height = 512
    num_channels = 14
    num_categories = 6

    # Load model
    model = deeplab_instance().construct(width, height, num_channels, num_categories)

    weights_filepath = Path("weights") / weights_name
    model.load_weights(weights_filepath)
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, width, height, num_channels)
    assert model.output_shape == (None, width, height, num_categories)

    # Load data

    subtiles = []
    labels = []
    for subtile_name in subtiles_names:
        download_test_data_from_s3(subtile_name)

        # Load test image
        subtile, label = load_and_normalize(Path("test_l2a_data") / subtile_name)
        assert subtile.shape == (width, height, num_channels)
        assert label.shape == (width, height)

        subtiles.append(subtile[np.newaxis])
        labels.append(label[np.newaxis])

    subtiles = np.vstack(subtiles)
    labels = np.vstack(labels)

    # Run prediction
    prediction = model.predict(subtiles)  # [np.newaxis])

    # Delete model
    tf.keras.backend.clear_session()

    assert prediction.shape == (len(subtiles_names), width, height, num_categories)

    # Compare predictions
    y_pred = np.argmax(prediction, axis=3)
    y_pred = y_pred.astype(np.uint8)
    assert np.array_equal(y_pred, labels), "Predictions are different."
