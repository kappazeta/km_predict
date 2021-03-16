import os
import random
import keras
import skimage.io as skio
import netCDF4 as nc
from keras.utils import np_utils
from PIL import Image
from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np


class DataGenerator(Sequence):
    def __init__(self, list_indices, path_input, batch_size, features, tile_size, num_classes, shuffle=True,
                 png_form=False):
        """ Initialization """
        self.path = path_input
        self.stds = [0.00085, 0.04, 0.037, 0.035, 0.034, 0.035, 0.033, 0.035, 0.034, 0.054, 0.025, 0.021, 0.0083]
        self.means = [0.0009, 0.02, 0.02, 0.02, 0.02, 0.041, 0.047, 0.045, 0.06, 0.03, 0.03, 0.022, 0.015]
        self.min_v = []
        self.max_v = []
        self.list_indices = list_indices
        self.total_length = len(self.list_indices)
        self.batch_size = batch_size
        self.png_form = png_form
        if png_form:
            self.features = ["TCI_R", "TCI_G", "TCI_B"]
        else:
            self.features = features
        self.tile_size = tile_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_indices) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch = [self.list_indices[k] for k in indexes]

        # Generate data
        X = self.__data_generation(batch)

        return X

    def set_std(self, stds):
        self.stds = stds

    def set_means(self, means):
        self.means = means

    def set_min(self, min_v):
        self.min_v = min_v

    def set_max(self, max_v):
        self.max_v = max_v

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_indices))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_indices_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, self.tile_size, self.tile_size, len(self.features)))
        y = np.empty((self.batch_size, self.tile_size, self.tile_size, self.num_classes), dtype=int)
        # Initialization
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    if self.png_form:
                        data_bands = [(np.asarray(root[f])) * 1 / 255 for i, f in
                                      enumerate(self.features)]
                    else:
                        data_bands = [(np.asarray(root[f]) - self.means[i]) / self.stds[i] for i, f in
                                      enumerate(self.features)]

                    data_bands = np.stack(data_bands)
                    data_bands = np.rollaxis(data_bands, 0, 3)
                    # data_bands = data_bands.reshape((self.dim[0], self.dim[1], len(self.features)))
                    X[i,] = data_bands

        return X

    def get_normal_par(self, list_indices_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        X = np.empty((len(list_indices_temp), self.tile_size, self.tile_size, len(self.features)))
        # Initialization
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    data_bands = [np.asarray(root[f]) for f in self.features]
                    data_bands = np.stack(data_bands)
                    data_bands = np.rollaxis(data_bands, 0, 3)
                    # data_bands = data_bands.reshape((self.dim[0], self.dim[1], len(self.features)))
                    X[i,] = data_bands

        stds_list = []
        means_list = []
        unique_list = []
        min_list = []
        max_list = []
        X_reshaped = np.reshape(X, (len(list_indices_temp) * self.tile_size * self.tile_size, len(self.features)))
        for j, class_curr in enumerate(self.features):
            # print(class_curr)
            std_array = np.std(X_reshaped[:, j])
            mean_array = np.mean(X_reshaped[:, j])
            unique = np.unique(X_reshaped[:, j])
            min_ar = np.min(X_reshaped[:, j])
            max_ar = np.max(X_reshaped[:, j])
            stds_list.append(std_array)
            means_list.append(mean_array)
            unique_list.append(unique)
            min_list.append(min_ar)
            max_list.append(max_ar)

        return stds_list, means_list, unique_list, min_list, max_list
