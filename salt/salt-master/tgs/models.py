from keras.layers import (
    Conv2D, MaxPooling2D, Dropout,
    Conv2DTranspose, concatenate,
    BatchNormalization, Concatenate,
    UpSampling2D, Input, add
)
from keras.models import Model
from tgs.layers import (
    SigmoidThreshold, ResidualBlock,
    RedBlock, BlueBlock
)
from tgs.config import TrainConfig as ReplacedTrainConfig


def unet_static(input_layer, start_neurons=16, **_):
    # 128 -> 64
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    # 64 -> 32
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    # 32 -> 16
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    # 16 -> 8
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    # uconv1 = Dropout(0.5)(uconv1)
    activation_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    return activation_layer


def conv_block(m, dim, acti, bn, res, do=0):
    n = Conv2D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n


def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
    return m


def unet_bn(input_layer, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
            dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False, **_):
    layers = level_block(input_layer, start_ch, depth, inc_rate, activation,
                         dropout, batchnorm, maxpool, upconv, residual)
    layers = Conv2D(out_ch, 1, activation='sigmoid')(layers)
    return layers


def unet(which='static', apply_threshold=False, input_shape=(128, 128, 1), unet_params=None):
    unet_params = unet_params or {}
    unet_map = {
        'static': unet_static,
        'bn': unet_bn
    }
    input_layer = Input(input_shape)
    output_layer = unet_map[which](input_layer, **unet_params)
    if apply_threshold:
        output_layer = SigmoidThreshold()(output_layer)

    return Model(input_layer, output_layer)


# Backward compatibility
TrainConfig = ReplacedTrainConfig
UNET = unet


class FusionNetComponents:

    @staticmethod
    def green_block(filters, kernel=3, activation="relu", dropout=None):
        """
        Single convolution block base
        Conv2D -- >> Activation -- >> BatchNorm
        """
        conv = Conv2D(filters, kernel, activation=activation, padding="same")
        conv = BatchNormalization()(conv)
        conv = Dropout(dropout)(conv) if dropout is not None else conv
        return conv

    def violet_block(self, filters, kernel=3, activation="relu", dropout=None):
        """
        Three convolution blocks from base
        Conv2D -- >> Conv2D -- >> BatchNorm
        """
        conv_1 = self.green_block(filters, kernel, activation, dropout)
        conv_2 = self.green_block(filters, kernel, activation, dropout)(conv_1)
        conv_3 = self.green_block(filters, kernel, activation, dropout)(conv_2)
        return conv_3

    def down_block(self, filters, kernel=3, activation="relu", dropout=None):
        block = self.green_block(filters, kernel, activation, dropout)
        block = self.violet_block(filters, kernel, activation, dropout)(block)


class FusionNet:

    """
    Keras implementation of FusionNet [https://arxiv.org/abs/1612.05360]
    """

    def __init__(self, input_shape=(128, 128, 1), start_filter=16):
        self.start_filter = start_filter
        # self.config = config
        self.input = Input(input_shape, name="input")
        self.output = self.build(self.input)
        self.model = Model(self.input, self.output)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def build(self, x):
        down_1 = ResidualBlock(self.start_filter, dropout=0.5, name="down_1")(x)
        pool_1 = BlueBlock(name="pool_1")(down_1)

        down_2 = ResidualBlock(self.start_filter * 2, dropout=0.5, name="down_2")(pool_1)
        pool_2 = BlueBlock(name="pool_2")(down_2)

        down_3 = ResidualBlock(self.start_filter * 4, dropout=0.5, name="down_3")(pool_2)
        pool_3 = BlueBlock(name="pool_3")(down_3)

        down_4 = ResidualBlock(self.start_filter * 8, dropout=0.5, name="down_4")(pool_3)
        pool_4 = BlueBlock(name="pool_4")(down_4)

        bridge = ResidualBlock(self.start_filter * 16, dropout=0.5, name="bridge")(pool_4)

        de_conv_4 = RedBlock(self.start_filter * 8, strides=2, name="de_conv_4")(bridge)
        res_4 = add([down_4, de_conv_4], name="res_4")
        up_4 = ResidualBlock(self.start_filter * 8, dropout=0.5, name="up_4")(res_4)

        de_conv_3 = RedBlock(self.start_filter * 4, strides=2, name="de_conv_3")(up_4)
        res_3 = add([down_3, de_conv_3], name="res_3")
        up_3 = ResidualBlock(self.start_filter * 4, dropout=0.5, name="up_3")(res_3)

        de_conv_2 = RedBlock(self.start_filter * 2, strides=2, name="de_conv_2")(up_3)
        res_2 = add([down_2, de_conv_2], name="res_2")
        up_2 = ResidualBlock(self.start_filter * 2, dropout=0.5, name="up_2")(res_2)

        de_conv_1 = RedBlock(self.start_filter, strides=2, name="de_conv_1")(up_2)
        res_1 = add([down_1, de_conv_1], name="res_1")
        up_1 = ResidualBlock(self.start_filter, dropout=0.5, name="up_1")(res_1)

        activation_layer = Conv2D(1, 1, padding="same", activation="sigmoid", name="activation_layer")(up_1)

        return activation_layer
