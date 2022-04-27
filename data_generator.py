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
            self.stds = [0.0013289578491821885, 0.036942049860954285, 0.03499632701277733, 0.032616470009088516, 0.03253139182925224, 0.033459462225437164, 0.03178662806749344, 0.03170452639460564, 0.032811496406793594, 0.03140449523925781, 0.053686607629060745, 0.024133026599884033, 0.021748408675193787, 0.014805539511144161]
            self.means = [0.0020673132967203856, 0.02537393383681774, 0.024809107184410095, 0.0262139905244112, 0.025707034394145012, 0.031577881425619125, 0.04240479692816734, 0.04607051610946655, 0.047901760786771774, 0.048266518861055374, 0.06299116462469101, 0.03296409174799919, 0.024804281070828438, 0.022374967113137245]
            self.min_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.max_v = [0.01107805036008358, 0.2983138859272003, 0.29272907972335815, 0.28123903274536133, 0.26414892077445984, 0.2527962028980255, 0.24835583567619324, 0.24779126048088074, 0.251224547624588, 0.24652475118637085, 0.25932708382606506, 0.23460745811462402, 0.2374914139509201, 0.12190432846546173]
        elif product_level == "L1C":
            self.stds = [0.023347536101937294, 0.024374037981033325, 0.023819683119654655, 0.02689390629529953, 0.026829229667782784, 0.027553776279091835, 0.029178336262702942, 0.02842315100133419, 0.03004441410303116, 0.016189342364668846, 0.004254127386957407, 0.022325700148940086, 0.018618354573845863]
            self.means = [0.031834159046411514, 0.02815815806388855, 0.02582935057580471, 0.025326654314994812, 0.028753258287906647, 0.03953897953033447, 0.044450148940086365, 0.04316269978880882, 0.04735855385661125, 0.018564173951745033, 0.002345206681638956, 0.03004465438425541, 0.021087026223540306]
            self.min_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.max_v = [0.19861142337322235, 0.29010453820228577, 0.3334401547908783, 0.3515373468399048, 0.269794762134552, 0.25995269417762756, 0.2672465145587921, 0.3791866898536682, 0.29629969596862793, 0.15063706040382385, 0.0642099678516388, 0.2062561959028244, 0.24827954173088074]
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
        self.indexes = []
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if len(self.list_indices) / self.batch_size:
            return int(np.floor(len(self.list_indices) / self.batch_size))
        else:
            return int(np.floor(len(self.list_indices) / self.batch_size)) + 1

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch = [self.list_indices[k] for k in indexes]

        # Generate data
        x = self.__data_generation(batch)

        return x

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
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_indices_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        x = np.empty((self.batch_size, self.tile_size, self.tile_size, len(self.features)))
        y = np.empty((self.batch_size, self.tile_size, self.tile_size, self.num_classes), dtype=int)
        # Initialization
        for i, file in enumerate(list_indices_temp):
            print(file)
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    if self.normalization == "minmax":
                        data_bands = [(np.asarray(root[f]) - self.min_v[i]) / (self.max_v[i] - self.min_v[i]) for i, f
                                      in enumerate(self.features)]
                    else:
                        data_bands = [(np.asarray(root[f]) - self.means[i]) / (self.stds[i]) for i, f
                                      in enumerate(self.features)]

                    data_bands = np.stack(data_bands)
                    data_bands = np.rollaxis(data_bands, 0, 3)
                    # data_bands = data_bands.reshape((self.dim[0], self.dim[1], len(self.features)))
                    x[i, ] = data_bands

        return x

    def get_normal_par(self, list_indices_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        x = np.empty((len(list_indices_temp), self.tile_size, self.tile_size, len(self.features)))
        # Initialization
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith('.nc'):
                with nc.Dataset(file, 'r') as root:
                    data_bands = [np.asarray(root[f]) for f in self.features]
                    data_bands = np.stack(data_bands)
                    data_bands = np.rollaxis(data_bands, 0, 3)
                    # data_bands = data_bands.reshape((self.dim[0], self.dim[1], len(self.features)))
                    x[i, ] = data_bands

        stds_list = []
        means_list = []
        unique_list = []
        min_list = []
        max_list = []
        x_reshaped = np.reshape(x, (len(list_indices_temp) * self.tile_size * self.tile_size, len(self.features)))
        for j, class_curr in enumerate(self.features):
            # print(class_curr)
            std_array = np.std(x_reshaped[:, j])
            mean_array = np.mean(x_reshaped[:, j])
            unique = np.unique(x_reshaped[:, j])
            min_ar = np.min(x_reshaped[:, j])
            max_ar = np.max(x_reshaped[:, j])
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


