"""Core module with Convolutional Discriminator related ops"""

import tensorflow as tf

from src.architectures import get_norm_layer, get_activation, \
                              DENSE_KERNEL_INITIALIZER, \
                              CONV_KERNEL_INITIALIZER
from src.callbacks.calls import LinearDecay

class ConvDiscriminator:
    """Utility class to build the discriminator.
    By the [paper](https://arxiv.org/abs/1703.10593v6) in section 4, the
    generative network architecture is adopt from [Johnson et al.](
                                                            https://arxiv.org/abs/1603.08155)
    """

    def __init__(self,
        input_shape=(256, 256, 3),
        dim=64,
        num_downsamplings=3,
        norm='instance_norm',
        lr_scheduler=LinearDecay(0.0001, 200, 100, 0.5)):

        self.norm = get_norm_layer(norm)
        self.input_shape = input_shape
        self.dim = dim
        self.dim_ = dim
        self.num_downsamplings = num_downsamplings
        self.lr_scheduler = lr_scheduler

        # build model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_scheduler, beta_1=self.lr_scheduler.beta_1)
        self.model = self.build()

    def build(self):
        x = inputs = tf.keras.Input(shape=self.input_shape)

        # 1
        x = tf.keras.layers.Conv2D(self.dim, 4, strides=2, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)        

        for _ in range(self.num_downsamplings - 1):
            self.dim = min(self.dim * 2, self.dim_ * 8)
            x = tf.keras.layers.Conv2D(self.dim, 4, strides=2, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER, use_bias=False)(x)
            x = self.norm()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # 2
        self.dim = min(self.dim * 2, self.dim_ * 8)
        x = tf.keras.layers.Conv2D(self.dim, 4, strides=1, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER, use_bias=False)(x)
        x = self.norm()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # 3
        x = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)

        model.compile(
            optimizer=self.optimizer
        )

        return model