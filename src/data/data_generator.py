"""Core module for data generator related operations"""
import tensorflow as tf
import os

from src import join

class Gen:

    def __init__(self):

        self.path = None
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
        image = tf.image.decode_png(image, 3) # fix channels to 3

        return image

    def data_generator(self):
        """Utility function to yield images

        Returns:
            Python generator
        """

        for image in self.all_images:
            try:

                image = self.parser_function(join(self.path, image))

                yield image
            except Exception as e:
                print(e)
                continue