from keras import backend as K
from keras.layers import (
    Conv2D, MaxPooling2D, Dropout,
    Conv2DTranspose,
    BatchNormalization,
    add, Layer, Input
)

sample_input = Input((128, 128, 1))


class SigmoidThreshold(Layer):
    """ Apply a threshold to sigmoid layer,
    and learn threshold value"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.threshold = self.add_weight(name='threshold',
                                         shape=(1, ),
                                         initializer='one',
                                         trainable=True)
        super().build(input_shape)

    def call(self, x, **kwargs):
        return K.cast(K.greater(x, self.threshold), x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape


class FusionBaseLayer:

    def __init__(self, filters, kernel_size=3, activation="relu",
                 dropout=None, strides=1, padding="same", **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size  # !
        self.activation = activation
        self.strides = strides  # !
        self.dropout = dropout
        self.padding = padding  # !
        self.params = dict(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            strides=strides,
        )

    def call(self, x,  **kwargs):
        return x

    def __call__(self, x, **kwargs):
        return self.call(x, **kwargs)


class GreenBlock(FusionBaseLayer):
    """
    Single convolution block base
    Conv2D -- >> Activation -- >> BatchNorm
    """

    def call(self, x, **kwargs):
        conv = Conv2D(**self.params)(x)
        conv = BatchNormalization()(conv)
        conv = Dropout(self.dropout)(conv) if self.dropout is not None else conv
        return conv


class VioletBlock(FusionBaseLayer):
    """
    Three convolution blocks from base block
    GreenBlock -- >> GreenBlock -- >> GreenBlock
    """

    def call(self, x, **kwargs):
        conv_1 = GreenBlock(**self.params, dropout=self.dropout)(x)
        conv_2 = GreenBlock(**self.params, dropout=self.dropout)(conv_1)
        conv_3 = GreenBlock(**self.params, dropout=self.dropout)(conv_2)
        return conv_3


class ResidualBlock(FusionBaseLayer):
    """
    The main residual block
    GreenBlock -- >> VioletBlock -- >> Skip Residual -- >> GreenBlock
    """

    def call(self, x, **kwargs):
        block_1 = GreenBlock(**self.params, dropout=self.dropout)(x)
        block_2 = VioletBlock(**self.params, dropout=self.dropout)(block_1)
        residual = add([block_1, block_2])
        block_3 = GreenBlock(**self.params, dropout=self.dropout)(residual)
        return block_3


class RedBlock(FusionBaseLayer):
    """
    Single deconvolution block for up-sampling
    Conv2DTranspose -- >> Activation -- >> BatchNorm
    """

    def call(self, x, **kwargs):
        conv = Conv2DTranspose(**self.params)(x)
        conv = BatchNormalization()(conv)
        conv = Dropout(self.dropout)(conv) if self.dropout is not None else conv
        return conv


class BlueBlock(MaxPooling2D):
    """
    Basically just MaxPooling for down-sampling
    GreenBlock -- >> VioletBlock -- >> Skip Residual -- >> GreenBlock
    """