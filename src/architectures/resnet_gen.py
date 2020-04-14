"""Core module with ResNet Generator related ops"""

import tensorflow as tf

from src.architectures import get_norm_layer, get_activation, \
                              DENSE_KERNEL_INITIALIZER, \
                              CONV_KERNEL_INITIALIZER
from src.callbacks.calls import LinearDecay

class ResNetGenerator:
    """Utility class to build the generator based on ResNet architecture.
    By the [paper](https://arxiv.org/abs/1703.10593v6) in section 4, the
    generative network architecture is adopt from [Johnson et al.](
                                                            https://arxiv.org/abs/1603.08155)
    """

    def __init__(self,
        input_shape=(256, 256, 3),
        output_channels=3,
        dim=64,
        num_downsamplings=2,
        num_blocks=9,
        norm='instance_norm',
        activation='relu'):

        self.norm = get_norm_layer(norm)
        self.activation = get_activation(activation)
        self.input_shape = input_shape
        self.output_channels = output_channels
        self.dim = dim
        self.num_downsamplings = num_downsamplings
        self.num_blocks = num_blocks

        # build model
        self.model = self.build()

    def residual_block(self, _input):
        dim = _input.shape[-1]

        x = tf.keras.layers.Conv2D(dim, 3, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER, use_bias=False)(_input)
        x = self.norm()(x)
        x = tf.keras.layers.Activation(self.activation)(x)

        x = tf.keras.layers.Conv2D(dim, 3, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER, use_bias=False)(x)
        x = self.norm()(x)

        return tf.keras.layers.add([x, _input])

    def build(self):
        x = inputs = tf.keras.Input(shape=self.input_shape)

        # 1
        x = tf.keras.layers.Conv2D(self.dim, 7, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER, use_bias=False)(x)
        x = self.norm()(x)
        x = tf.keras.layers.Activation(self.activation)(x)

        # 2
        for _ in range(self.num_downsamplings):
            self.dim *= 2
            x = tf.keras.layers.Conv2D(self.dim, 3, strides=2, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER, use_bias=False)(x)
            x = self.norm()(x)
            x = tf.keras.layers.Activation(self.activation)(x)

        # 3
        for _ in range(self.num_blocks):
            x = self.residual_block(x)

        # 4
        for _ in range(self.num_downsamplings):
            self.dim //= 2
            x = tf.keras.layers.Conv2DTranspose(self.dim, 3, strides=2, padding='same', kernel_initiliazer=CONV_KERNEL_INITIALIZER, use_bias=False)(x)
            x = self.norm()(x)            
            x = tf.keras.layers.Activation(self.activation)(x)

        # 5
        x = tf.keras.layers.Conv2D(self.output_channels, 7, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
        x = tf.tanh(x)

        return tf.keras.Model(inputs=inputs, outputs=x)