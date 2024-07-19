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
from platform import architecture
import skimage.io as skio
import netCDF4 as nc
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.utils import Sequence
import numpy as np


class DataGenerator(Sequence):
    def __init__(
        self,
        list_indices,
        path_input,
        architecture,
        batch_size,
        features,
        tile_size,
        num_classes,
        product_level,
        offsets,
        shuffle=True,
        png_form=False,
    ):
        """Initialization"""
        self.path = path_input
        if product_level == "L2A":
            if architecture == "DeepLabv3Plus":
                self.stds = [
                    0.0012938244035467505,
                    0.04729962348937988,
                    0.04480421543121338,
                    0.04136919602751732,
                    0.040766045451164246,
                    0.04161246493458748,
                    0.03878653794527054,
                    0.037818487733602524,
                    0.03898065164685249,
                    0.03693762049078941,
                    0.06387823820114136,
                    0.024207308888435364,
                    0.021626926958560944,
                    0.016903972253203392,
                ]
                self.means = [
                    0.002305045025423169,
                    0.03792252019047737,
                    0.036831602454185486,
                    0.036885615438222885,
                    0.036290477961301804,
                    0.041195500642061234,
                    0.04801696166396141,
                    0.05018390342593193,
                    0.05178681015968323,
                    0.05127640441060066,
                    0.07079055160284042,
                    0.0315968282520771,
                    0.025250067934393883,
                    0.022059641778469086,
                ]
                self.min_v = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                self.max_v = [
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
            elif architecture == "Unet":
                self.stds = [
                    0.000845261150971055,
                    0.041299913078546524,
                    0.039003968238830566,
                    0.03623047098517418,
                    0.03549625352025032,
                    0.036688629537820816,
                    0.03459501639008522,
                    0.03538798168301582,
                    0.03376650810241699,
                    0.060521017760038376,
                    0.02391253225505352,
                    0.021194500848650932,
                    0.0115975895896554,
                ]
                self.means = [
                    0.0016591775929555297,
                    0.029182588681578636,
                    0.02786719985306263,
                    0.028459852561354637,
                    0.02683664672076702,
                    0.032926056534051895,
                    0.044384393841028214,
                    0.05008489266037941,
                    0.05016825720667839,
                    0.06914236396551132,
                    0.03178902342915535,
                    0.023803194984793663,
                    0.020374732092022896,
                ]
                self.min_v = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                self.max_v = [
                    0.003906309604644775,
                    0.2833447754383087,
                    0.27905699610710144,
                    0.2668497860431671,
                    0.2558480203151703,
                    0.250980406999588,
                    0.24597543478012085,
                    0.24182498455047607,
                    0.24206912517547607,
                    0.2535133957862854,
                    0.2232242375612259,
                    0.23524834215641022,
                    0.07557793706655502,
                ]  # l1c_v67
        elif product_level == "L1C":
            if architecture == "DeepLabv3Plus":
                self.stds = [
                    0.02871280163526535,
                    0.029620453715324402,
                    0.028293564915657043,
                    0.031824689358472824,
                    0.03180317208170891,
                    0.031496092677116394,
                    0.03233364596962929,
                    0.03155158832669258,
                    0.032838210463523865,
                    0.019741980358958244,
                    0.006373442243784666,
                    0.021687006577849388,
                    0.017932893708348274,
                ]
                self.means = [
                    0.04005281999707222,
                    0.03640437498688698,
                    0.033134810626506805,
                    0.03343692049384117,
                    0.0359625518321991,
                    0.043630972504615784,
                    0.047440607100725174,
                    0.04572567716240883,
                    0.0494818389415741,
                    0.021328648552298546,
                    0.0033179752063006163,
                    0.0288138035684824,
                    0.02113727293908596,
                ]
                self.min_v = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                self.max_v = [
                    0.2065003365278244,
                    0.3510643243789673,
                    0.3327534794807434,
                    0.3624475598335266,
                    0.29547569155693054,
                    0.29562827944755554,
                    0.29103532433509827,
                    0.3791866898536682,
                    0.2850232720375061,
                    0.18451209366321564,
                    0.0799115002155304,
                    0.24786755442619324,
                    0.24827954173088074,
                ]
            elif architecture == "Unet":
                self.stds = [
                    0.0246,
                    0.025,
                    0.0251,
                    0.0278,
                    0.0277,
                    0.0284,
                    0.030,
                    0.0292,
                    0.0309,
                    0.0168,
                    0.0045,
                    0.0213,
                    0.0175,
                ]
                self.means = [
                    0.0321,
                    0.0284,
                    0.0259,
                    0.0246,
                    0.0282,
                    0.0400,
                    0.0453,
                    0.0439,
                    0.0484,
                    0.0188,
                    0.0024,
                    0.0288,
                    0.0197,
                ]
                self.min_v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                self.max_v = [
                    0.21,
                    0.326,
                    0.264,
                    0.298,
                    0.245,
                    0.258,
                    0.266,
                    0.364,
                    0.259,
                    0.165,
                    0.055,
                    0.203,
                    0.208,
                ]
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
        self.offsets = offsets
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
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

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
        x = np.empty(
            (self.batch_size, self.tile_size, self.tile_size, len(self.features))
        )
        y = np.empty(
            (self.batch_size, self.tile_size, self.tile_size, self.num_classes),
            dtype=int,
        )

        self.max_v = list(np.array(self.max_v) + self.offsets / 65535)
        # Initialization
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith(".nc"):
                with nc.Dataset(file, "r") as root:
                    if self.normalization == "minmax":
                        # Take data bands from the NetCDF file, add offsets from S2 product metadata, scale pixel values by range.
                        data_bands = np.asarray(
                            [np.asarray(root[f]) for i, f in enumerate(self.features)]
                        )
                        data_bands = data_bands + (
                            np.reshape(self.offsets, (len(self.features), 1, 1)) / 65535
                        )
                        data_bands = [
                            (data_bands[i] - self.min_v[i])
                            / (self.max_v[i] - self.min_v[i])
                            for i, f in enumerate(self.features)
                        ]
                    else:
                        data_bands = [
                            ((np.asarray(root[f])) - self.means[i]) / (self.stds[i])
                            for i, f in enumerate(self.features)
                        ]

                    data_bands = np.stack(data_bands)
                    data_bands = np.rollaxis(data_bands, 0, 3)
                    data_bands[data_bands < 0] = 0
                    x[i,] = data_bands

        return x

    def get_normal_par(self, list_indices_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        x = np.empty(
            (len(list_indices_temp), self.tile_size, self.tile_size, len(self.features))
        )
        # Initialization
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith(".nc"):
                with nc.Dataset(file, "r") as root:
                    data_bands = [np.asarray(root[f]) for f in self.features]
                    data_bands = np.stack(data_bands)
                    data_bands = np.rollaxis(data_bands, 0, 3)
                    x[i,] = data_bands

        stds_list = []
        means_list = []
        unique_list = []
        min_list = []
        max_list = []
        x_reshaped = np.reshape(
            x,
            (
                len(list_indices_temp) * self.tile_size * self.tile_size,
                len(self.features),
            ),
        )
        for j, class_curr in enumerate(self.features):
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
        y = np.zeros(
            (len(self.list_indices), self.tile_size, self.tile_size, self.num_classes),
            dtype=np.float32,
        )
        # Initialization
        for i, file in enumerate(self.list_indices):
            if os.path.isfile(file) and file.endswith(".nc"):
                with nc.Dataset(file, "r") as root:
                    try:
                        sen2cor = np.asarray(root["SCL"])
                        y[i] = np_utils.to_categorical(sen2cor, self.num_classes)
                    except:
                        print("Sen2Cor for confusion " + file + " not found")
        return y

    def store_orig(self, list_indices_temp, path_prediction):
        """Save labels to folder"""
        for i, file in enumerate(list_indices_temp):
            if os.path.isfile(file) and file.endswith(".nc"):
                with nc.Dataset(file, "r") as root:
                    y = np.empty(
                        (self.tile_size, self.tile_size, self.num_classes), dtype=int
                    )
                    data_bands = [
                        np.asarray(root[f])
                        for i, f in enumerate(["TCI_R", "TCI_G", "TCI_B"])
                    ]
                    data_bands = np.stack(data_bands)
                    data_bands = np.rollaxis(data_bands, 0, 3)

                    file_name = file.split(".")[0].split("/")[-1]

                    if not os.path.exists(path_prediction + "/" + file_name):
                        os.mkdir(path_prediction + "/" + file_name)

                    data_bands = data_bands.astype(np.uint8)
                    skio.imsave(
                        path_prediction + "/" + file_name + "/orig.png", data_bands
                    )
