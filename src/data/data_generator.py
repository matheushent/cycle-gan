"""Core module for data generator related operations"""
import tensorflow as tf
import sys
import os

from src import join

class Gen:

    def __init__(self):

        self.dtype = tf.uint8
        self.all_images = None

    def parser_function(self, path):
        """Utility function to parser each path

        Args:
            path (str): Image path containing images

        Returns:
            tf.Tensor: 3D-Tensor with shape (H, W, 3)
        """

        image = tf.io.read_file(path)
        image = tf.cond(
            tf.image.is_jpeg(image),
            lambda: tf.image.decode_jpeg(image, 3), # fix channels to 3
            lambda: tf.image.decode_png(image, 3)
        )

        return image

    def data_generator(self):
        """Utility function to yield images

        Returns:
            Python generator
        """

        for image in self.all_images:
            try:

                image = self.parser_function(image)

                yield image
            except Exception as e:
                print(e)
                continue