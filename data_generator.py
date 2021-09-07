# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# KappaMask model and metrics.
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
import skimage.io as skio
import netCDF4 as nc
from keras.utils import np_utils
from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np


class DataGenerator(Sequence):
    def __init__(self, list_indices, path_input, batch_size, features, tile_size, num_classes, product_level,
                 shuffle=True, png_form=False):
        """ Initialization """
        self.path = path_input
        if product_level == "L2A":
            self.stds = [0.000845261150971055, 0.041299913078546524, 0.039003968238830566, 0.03623047098517418, 0.03549625352025032, 0.036688629537820816, 0.03459501639008522, 0.03538798168301582, 0.03376650810241699, 0.060521017760038376, 0.02391253225505352, 0.021194500848650932, 0.0115975895896554]
            self.means = [0.0016591775929555297, 0.029182588681578636, 0.02786719985306263, 0.028459852561354637, 0.02683664672076702, 0.032926056534051895, 0.044384393841028214, 0.05008489266037941, 0.05016825720667839, 0.06914236396551132, 0.03178902342915535, 0.023803194984793663, 0.020374732092022896]
            self.min_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.max_v = [0.003906309604644775, 0.2833447754383087, 0.27905699610710144, 0.2668497860431671, 0.2558480203151703, 0.250980406999588, 0.24597543478012085, 0.24182498455047607, 0.24206912517547607, 0.2535133957862854, 0.2232242375612259, 0.23524834215641022, 0.07557793706655502]
        elif product_level == "L1C":
            self.stds = [0.0246, 0.025, 0.0251, 0.0278, 0.0277, 0.0284, 0.030, 0.0292, 0.0309, 0.0168, 0.0045, 0.0213,
                         0.0175]
            self.means = [0.0321, 0.0284, 0.0259, 0.0246, 0.0282, 0.0400, 0.0453, 0.0439, 0.0484, 0.0188, 0.0024,
                          0.0288, 0.0197]
            self.min_v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.max_v = [0.21, 0.326, 0.264, 0.298, 0.245, 0.258, 0.266, 0.364, 0.259, 0.165, 0.055, 0.203, 0.208]
        self.normalization = "minmax"
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
                    if self.normalization == "minmax":
                        data_bands = [(np.asarray(root[f]) - self.min_v[i]) / (self.max_v[i] - self.min_v[i]) for i, f
                                      in
                                      enumerate(self.features)]
                    else:
                        data_bands = [(np.asarray(root[f]) - self.means[i]) / (self.stds[i]) for i, f
                                      in enumerate(self.features)]

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

    def get_sen2cor(self):
        y = np.zeros((len(self.list_indices), self.tile_size, self.tile_size, self.num_classes), dtype=np.float32)
        # Initialization
        for i, file in enumerate(self.list_indices):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    try:
                        sen2cor = np.asarray(root["SCL"])
                        y[i] = np_utils.to_categorical(sen2cor, self.num_classes)
                    except:
                        print("Sen2Cor for confusion " + file + " not found")
        return y

    def store_orig(self, list_indices_temp, path_prediction):
        """Save labels to folder"""
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    y = np.empty((self.tile_size, self.tile_size, self.num_classes), dtype=int)
                    data_bands = [np.asarray(root[f])
                                  for i, f in enumerate(["TCI_R", "TCI_G", "TCI_B"])]
                    data_bands = np.stack(data_bands)
                    data_bands = np.rollaxis(data_bands, 0, 3)

                    file_name = file.split(".")[0].split("/")[-1]

                    if not os.path.exists(path_prediction + "/" + file_name):
                        os.mkdir(path_prediction + "/" + file_name)

                    data_bands = data_bands.astype(np.uint8)
                    skio.imsave(path_prediction + "/" + file_name + "/orig.png", data_bands)


