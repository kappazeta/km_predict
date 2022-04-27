# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# KappaMask model architectures.
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

import tensorflow as tf
from model import CMModel


class Unet(CMModel):
    """
    Unet
    """

    def __init__(self):
        super(Unet, self).__init__("CMP.M")

    def construct(self, width, height, num_channels, num_categories, pretrained_weights=False):
        """
        Construct the model.
        :param width: Width of a single sample (must be an odd number).
        :param height: Height of a single sample (must be an odd number).
        :param num_channels: Number of features used.
        :param num_categories: Number of output classes.
        """
        # For symmetrical neighbourhood, width and height must be odd numbers.
        self.input_shape = (width, height, num_channels)
        self.output_shape = (num_categories,)

        with tf.name_scope("Model"):
            inputs = tf.keras.layers.Input(self.input_shape, name='input')

            conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)

            conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)

            conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)

            conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            drop4 = tf.keras.layers.Dropout(0.5)(conv4)
            pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(drop4)

            conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
            conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            drop5 = tf.keras.layers.Dropout(0.5)(conv5)

            up6 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
            merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
            conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
            conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

            up7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
            merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
            conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
            conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

            up8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
            merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
            conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

            up9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
            merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)

            conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
            conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv10 = tf.keras.layers.Conv2D(num_categories, (1, 1), activation='softmax')(conv9)

            self.model = tf.keras.Model(inputs, conv10)

            self.model.summary()

            if pretrained_weights:
                self.model.load_weights(pretrained_weights)

            return self.model

class DeepLab(CMModel):
    """
    DeepLab v3
    """

    def __init__(self):
        super(DeepLab, self).__init__("CMP.M")

    def construct(self, width, height, num_channels, num_categories, pretrained_weights=False):
        """
        Construct the model.
        :param width: Width of a single sample (must be an odd number).
        :param height: Height of a single sample (must be an odd number).
        :param num_channels: Number of features used.
        :param num_categories: Number of output classes.
        """
        # For symmetrical neighbourhood, width and height must be odd numbers.
        self.input_shape = (width,height, num_channels)
        self.output_shape = (num_categories,)
        n_filters = 64

        with tf.name_scope('Model'):
            inputs = tf.keras.layers.Input(self.input_shape, name='input')
            x_padding = tf.keras.layers.ZeroPadding2D(padding = (2,2))(inputs)
            xception = tf.keras.applications.Xception(include_top = False, input_tensor = x_padding, weights = None)
            x = xception.get_layer('block13_sepconv2_bn').output
            x = self.ASPP(x)

            input_a = tf.keras.layers.UpSampling2D(size = (width // 4 // x.shape[1], height // 4 // x.shape[2]),
            interpolation = 'bilinear')(x)
            input_b = xception.get_layer('block3_sepconv2_bn').output
            input_b = self.convolutional_block(input_b, num_filters = 48, kernel_size = 1)
            
            x = tf.keras.layers.Concatenate(axis = -1)([input_a, input_b])
            x = self.convolutional_block(x)
            x = self.convolutional_block(x)
            x = tf.keras.layers.UpSampling2D(size = (width // x.shape[1], height // x.shape[2]), interpolation =
            'bilinear')(x)
            outputs = tf.keras.layers.Conv2D(num_categories, activation = 'softmax', kernel_size = (1,1), padding = 'same')(x)

            self.model = tf.keras.Model(inputs, outputs)
            if pretrained_weights:
                self.model.load_weights(pretrained_weights)

            self.model.summary()
            
            return self.model

    def convolutional_block(self, input_, num_filters = 256, kernel_size = 3, dilation_rate = 1, padding = 'same',
    use_bias = False):
        conv_x = tf.keras.layers.Conv2D(num_filters, kernel_size = kernel_size, dilation_rate = dilation_rate, padding =
        padding, use_bias = use_bias, kernel_initializer = 'he_normal')(input_)
        x = tf.keras.layers.BatchNormalization()(conv_x)
        return tf.nn.relu(x)

    def ASPP(self, input_):
        dims = input_.shape
        x = tf.keras.layers.AveragePooling2D(pool_size = (dims[-3], dims[-2]))(input_)
        x = self.convolutional_block(x, kernel_size = 1, use_bias = True)
        out_pool = tf.keras.layers.UpSampling2D(size = (dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation =
        'bilinear')(x)

        out_1 = self.convolutional_block(input_, kernel_size = 1, dilation_rate = 1)
        out_6 = self.convolutional_block(input_, kernel_size = 3, dilation_rate = 6)
        out_12 = self.convolutional_block(input_, kernel_size = 3, dilation_rate = 12)
        out_18 = self.convolutional_block(input_, kernel_size = 3, dilation_rate = 18)
        
        x = tf.keras.layers.Concatenate(axis = -1)([out_pool, out_1, out_6, out_12, out_18])
        output = self.convolutional_block(x, kernel_size = 1)

        return output



ARCH_MAP = {
    "Unet": Unet,
    "DeepLab" : DeepLab,
}
