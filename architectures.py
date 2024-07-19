# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# Copyright 2020 KappaZeta Ltd.
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
# See t
# he License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from model import CMModel


class CustomPad(tf.keras.layers.Layer):
    def __init__(self, stride, kernel_size, dil_size=1):
        super(CustomPad, self).__init__()
        self.stride = stride
        self.filter_h = kernel_size + (kernel_size - 1) * (dil_size - 1)
        self.filter_w = kernel_size + (kernel_size - 1) * (dil_size - 1)

    def build(self, input_shape):
        input_h = input_shape[1]
        input_w = input_shape[2]

        if input_h % self.stride == 0:
            pad_along_height = max((self.filter_h - self.stride), 0)
        else:
            pad_along_height = max(self.filter_h - (input_h % self.stride), 0)
        if input_w % self.stride == 0:
            pad_along_width = max((self.filter_w - self.stride), 0)
        else:
            pad_along_width = max(self.filter_w - (input_w % self.stride), 0)

        self.pad_top = pad_along_height // 2  # amount of zero padding on the top
        self.pad_bottom = (
            pad_along_height - self.pad_top
        )  # amount of zero padding on the bottom
        self.pad_left = pad_along_width // 2  # amount of zero padding on the left
        self.pad_right = (
            pad_along_width - self.pad_left
        )  # amount of zero padding on the right

    def call(self, inputs):
        # print(self.pad_top, self.pad_bottom, self.pad_left, self.pad_right)
        return tf.pad(
            inputs,
            (
                (0, 0),
                (self.pad_left, self.pad_right),
                (self.pad_top, self.pad_bottom),
                (0, 0),
            ),
            "SYMMETRIC",
        )


class XCeption(tf.keras.Model):
    def __init__(self, input_tensor=None, input_shape=None):
        super(XCeption, self).__init__()

    def conv_bn(self, x, filters, kernel_size, strides=1):
        x = CustomPad(stride=strides, kernel_size=kernel_size)(x)
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

    def sep_bn(self, x, filters, kernel_size, strides=1):
        x = CustomPad(stride=strides, kernel_size=kernel_size)(x)
        x = tf.keras.layers.SeparableConv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

    def entry_flow(self, x):
        x = self.conv_bn(x, filters=32, kernel_size=3, strides=2)
        x = tf.keras.layers.ReLU()(x)
        x = self.conv_bn(x, filters=64, kernel_size=3, strides=1)
        tensor = tf.keras.layers.ReLU()(x)

        x = self.sep_bn(tensor, filters=128, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        x = self.sep_bn(x, filters=128, kernel_size=3)
        x = CustomPad(kernel_size=3, stride=2)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid")(x)

        tensor = self.conv_bn(tensor, filters=128, kernel_size=1, strides=2)
        x = tf.keras.layers.Add()([tensor, x])

        x = tf.keras.layers.ReLU()(x)
        x = self.sep_bn(x, filters=256, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        x = self.sep_bn(x, filters=256, kernel_size=3)
        x = CustomPad(kernel_size=3, stride=2)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid")(x)

        tensor = self.conv_bn(tensor, filters=256, kernel_size=1, strides=2)
        x = tf.keras.layers.Add()([tensor, x])

        x = tf.keras.layers.ReLU()(x)
        x = self.sep_bn(x, filters=728, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        x = self.sep_bn(x, filters=728, kernel_size=3)
        x = CustomPad(kernel_size=3, stride=2)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid")(x)

        tensor = self.conv_bn(tensor, filters=728, kernel_size=1, strides=2)
        x = tf.keras.layers.Add()([tensor, x])
        return x

    def middle_flow(self, tensor):
        for _ in range(8):
            x = tf.keras.layers.ReLU()(tensor)
            x = self.sep_bn(x, filters=728, kernel_size=3)
            x = tf.keras.layers.ReLU()(x)
            x = self.sep_bn(x, filters=728, kernel_size=3)
            x = tf.keras.layers.ReLU()(x)
            x = self.sep_bn(x, filters=728, kernel_size=3)
            x = tf.keras.layers.ReLU()(x)
            tensor = tf.keras.layers.Add()([tensor, x])

        return tensor

    def exit_flow(self, tensor):
        x = tf.keras.layers.ReLU()(tensor)
        x = self.sep_bn(x, filters=728, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        x = self.sep_bn(x, filters=1024, kernel_size=3)
        x = CustomPad(kernel_size=3, stride=2)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid")(x)

        tensor = self.conv_bn(tensor, filters=1024, kernel_size=1, strides=2)
        x = tf.keras.layers.Add()([tensor, x])

        x = self.sep_bn(x, filters=1536, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        x = self.sep_bn(x, filters=2048, kernel_size=3)
        x = tf.keras.layers.GlobalAvgPool2D()(x)

        x = tf.keras.layers.Dense(units=1000, activation="softmax")(x)

        return x

    def call(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        output = self.exit_flow(x)
        return output

    def build_graph(self, input_tensor):
        model = tf.keras.Model(inputs=input_tensor, outputs=self.call(input_tensor))
        return model


class Unet(CMModel):
    """
    Unet
    """

    def __init__(self):
        super(Unet, self).__init__("Unet")

    def construct(
        self,
        width,
        height,
        num_channels,
        num_categories,
        layers=False,
        units=False,
        pretrained_weights=False,
    ):
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
        if units:
            n_filters = units
        else:
            n_filters = 64
        growth_factor = 2

        with tf.name_scope("Model"):
            inputs = tf.keras.layers.Input(self.input_shape, name="input")

            conv1 = tf.keras.layers.Conv2D(
                n_filters,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(inputs)
            conv1 = tf.keras.layers.Conv2D(
                n_filters,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(conv1)
            pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)
            n_filters *= growth_factor

            conv2 = tf.keras.layers.Conv2D(
                n_filters,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(pool1)
            conv2 = tf.keras.layers.Conv2D(
                n_filters,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(conv2)
            pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)
            n_filters *= growth_factor

            conv3 = tf.keras.layers.Conv2D(
                n_filters,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(pool2)
            conv3 = tf.keras.layers.Conv2D(
                n_filters,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(conv3)
            pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)
            n_filters *= growth_factor

            if layers == 5 or layers == False:
                conv4 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(pool3)
                conv4 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv4)
                drop4 = tf.keras.layers.Dropout(0.5)(conv4)
                pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(drop4)
                n_filters *= growth_factor

                conv_middle = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(pool4)
                conv_middle = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv_middle)
                drop_middle = tf.keras.layers.Dropout(0.5)(conv_middle)
                n_filters //= growth_factor

                up8 = tf.keras.layers.Conv2D(
                    n_filters,
                    2,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(tf.keras.layers.UpSampling2D(size=(2, 2))(drop_middle))
                merge8 = tf.keras.layers.concatenate([drop4, up8], axis=3)
                conv8 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(merge8)
                conv8 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv8)
                n_filters //= growth_factor
            elif layers == 6:
                conv4 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(pool3)
                conv4 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv4)
                pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv4)
                n_filters *= growth_factor

                conv5 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(pool4)
                conv5 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv5)
                drop5 = tf.keras.layers.Dropout(0.5)(conv5)
                pool5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(drop5)
                n_filters *= growth_factor

                conv_middle = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(pool5)
                conv_middle = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv_middle)
                drop_middle = tf.keras.layers.Dropout(0.5)(conv_middle)
                n_filters //= growth_factor

                up7 = tf.keras.layers.Conv2D(
                    n_filters,
                    2,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(tf.keras.layers.UpSampling2D(size=(2, 2))(drop_middle))
                merge7 = tf.keras.layers.concatenate([drop5, up7], axis=3)
                conv7 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(merge7)
                conv7 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv7)
                n_filters //= growth_factor

                up8 = tf.keras.layers.Conv2D(
                    n_filters,
                    2,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
                merge8 = tf.keras.layers.concatenate([conv4, up8], axis=3)
                conv8 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(merge8)
                conv8 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv8)
                n_filters //= growth_factor
            elif layers == 7:
                conv4 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(pool3)
                conv4 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv4)
                pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv4)
                n_filters *= growth_factor

                conv5 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(pool4)
                conv5 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv5)
                pool5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv5)
                n_filters *= growth_factor

                conv6 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(pool5)
                conv6 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv6)
                drop6 = tf.keras.layers.Dropout(0.5)(conv6)
                pool6 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(drop6)
                n_filters *= growth_factor

                conv_middle = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(pool6)
                conv_middle = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv_middle)
                drop_middle = tf.keras.layers.Dropout(0.5)(conv_middle)
                n_filters //= growth_factor

                up7 = tf.keras.layers.Conv2D(
                    n_filters,
                    2,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(tf.keras.layers.UpSampling2D(size=(2, 2))(drop_middle))
                merge7 = tf.keras.layers.concatenate([drop6, up7], axis=3)
                conv7 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(merge7)
                conv7 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv7)
                n_filters //= growth_factor

                up8_1 = tf.keras.layers.Conv2D(
                    n_filters,
                    2,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
                merge8_1 = tf.keras.layers.concatenate([conv5, up8_1], axis=3)
                conv8_1 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(merge8_1)
                conv8_1 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv8_1)
                n_filters //= growth_factor

                up8 = tf.keras.layers.Conv2D(
                    n_filters,
                    2,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(tf.keras.layers.UpSampling2D(size=(2, 2))(conv8_1))
                merge8 = tf.keras.layers.concatenate([conv4, up8], axis=3)
                conv8 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(merge8)
                conv8 = tf.keras.layers.Conv2D(
                    n_filters,
                    3,
                    activation="relu",
                    padding="same",
                    kernel_initializer="he_normal",
                )(conv8)
                n_filters //= growth_factor

            up9 = tf.keras.layers.Conv2D(
                n_filters,
                2,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
            merge9 = tf.keras.layers.concatenate([conv3, up9], axis=3)
            conv9 = tf.keras.layers.Conv2D(
                n_filters,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(merge9)
            conv9 = tf.keras.layers.Conv2D(
                n_filters,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(conv9)
            n_filters //= growth_factor

            up10 = tf.keras.layers.Conv2D(
                n_filters,
                2,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(tf.keras.layers.UpSampling2D(size=(2, 2))(conv9))
            merge10 = tf.keras.layers.concatenate([conv2, up10], axis=3)

            conv10 = tf.keras.layers.Conv2D(
                n_filters,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(merge10)
            conv10 = tf.keras.layers.Conv2D(
                n_filters,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(conv10)
            n_filters //= growth_factor

            up11 = tf.keras.layers.Conv2D(
                n_filters,
                2,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(tf.keras.layers.UpSampling2D(size=(2, 2))(conv10))
            merge11 = tf.keras.layers.concatenate([conv1, up11], axis=3)

            conv11 = tf.keras.layers.Conv2D(
                n_filters,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(merge11)
            conv11 = tf.keras.layers.Conv2D(
                n_filters,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )(conv11)
            conv11 = tf.keras.layers.Conv2D(
                2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
            )(conv11)
            conv12 = tf.keras.layers.Conv2D(
                num_categories, (1, 1), activation="sigmoid"
            )(conv11)

            self.model = tf.keras.Model(inputs, conv12)

            # self.model.summary()

            if pretrained_weights:
                self.model.load_weights(pretrained_weights)

            return self.model


class DeepLabv3Plus(CMModel):
    """
    DeepLabv3+ Aligned
    """

    def __init__(self):
        super(DeepLabv3Plus, self).__init__("DeepLabv3Plus")

    def construct(
        self,
        width,
        height,
        num_channels,
        num_categories,
        layers=False,
        units=False,
        pretrained_weights=False,
    ):
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
            inputs = tf.keras.layers.Input(self.input_shape, name="input")
            extractor = XCeption().build_graph(inputs)
            x = extractor.get_layer("batch_normalization_36").output

            x = self.ASPP(x)

            input_a = tf.keras.layers.UpSampling2D(
                size=(width // 4 // x.shape[1], height // 4 // x.shape[2]),
                interpolation="bilinear",
            )(x)

            input_b = extractor.get_layer("batch_normalization_6").output
            input_b = self.convolutional_block(input_b, num_filters=48, kernel_size=1)

            x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
            x = self.convolutional_block(x)
            x = self.convolutional_block(x)
            x = tf.keras.layers.UpSampling2D(
                size=(width // x.shape[1], height // x.shape[2]),
                interpolation="bilinear",
            )(x)
            outputs = tf.keras.layers.Conv2D(
                num_categories,
                activation="softmax",
                kernel_size=(1, 1),
                padding="valid",
            )(x)

            self.model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
            if pretrained_weights:
                self.model.load_weights(pretrained_weights)

            self.model.summary()

            return self.model

    def convolutional_block(
        self,
        input_,
        num_filters=256,
        kernel_size=3,
        dilation_rate=1,
        padding="valid",
        strides=1,
        use_bias=False,
    ):
        pad_x = CustomPad(
            stride=strides, kernel_size=kernel_size, dil_size=dilation_rate
        )(input_)
        conv_x = tf.keras.layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer="he_normal",
        )(pad_x)
        x = tf.keras.layers.BatchNormalization()(conv_x)
        return tf.nn.relu(x)

    def ASPP(self, input_):
        dims = input_.shape
        x = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(input_)
        x = self.convolutional_block(x, kernel_size=1, use_bias=True)
        out_pool = tf.keras.layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
            interpolation="bilinear",
        )(x)

        out_1 = self.convolutional_block(input_, kernel_size=1, dilation_rate=1)
        out_6 = self.convolutional_block(input_, kernel_size=3, dilation_rate=6)
        out_12 = self.convolutional_block(input_, kernel_size=3, dilation_rate=12)
        out_18 = self.convolutional_block(input_, kernel_size=3, dilation_rate=18)

        x = tf.keras.layers.Concatenate(axis=-1)(
            [out_pool, out_1, out_6, out_12, out_18]
        )
        output = self.convolutional_block(x, kernel_size=1)

        return output


ARCH_MAP = {"Unet": Unet, "DeepLabv3Plus": DeepLabv3Plus}
