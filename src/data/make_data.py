"""Core module for dataset building related operations"""
import tensorflow as tf

import multiprocessing

from imutils import paths

from src.data.data_generator import Gen
from src import join

class MakeDataset:
    """
    Args:
        C (object): Config class
        path (str): Folder path containing images, e.g. './datasets/monet2photo'
        training (bool): Boolean to indicate training or testing
    """

    def __init__(self, C, path, training, shuffle=True, drop_remainder=True, repeat=False):
        
        self.C = C
        self.path = path
        self.training = training
        self.map_fn = MakeDataset.get_map_function(self.training, self.C)
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.repeat = repeat
        self.gen_a = Gen()
        self.gen_b = Gen()

        if self.shuffle:
            self.shuffle_buffer_size = max(self.C.batch_size * 128, 2048)

    @staticmethod
    def get_map_function(training, C):
        """Utility function to get the map function according
        to mode (train ou eval)

        Args:
            training (bool): Boolean to indicate training or testing
            

        Returns:
            tf.function
        """

        if training:

            @tf.function
            def _map(image):
                """Utility function to act as map

                Args:
                    image (tf.Tensor): 3D-Tensor with shape (H, W, 3)

                Returns:
                    tf.Tendor: 3D-Tensor with shape (H, W, 3)
                """

                image = tf.image.random_flip_left_right(image, seed=C.seed)
                image = tf.image.resize(image, (C.load_size, C.load_size))
                image = tf.image.random_crop(image, (C.crop_size, C.crop_size, tf.shape(image)[-1]), seed=C.seed)
                image = tf.clip_by_value(image, 0, 255) / 255.0
                image = image * 2 - 1

                return image
        else:

            @tf.function
            def _map(image):
                """Utility function to act as map

                Args:
                    image (tf.Tensor): 3D-Tensor with shape (H, W, 3)

                Returns:
                    tf.Tendor: 3D-Tensor with shape (H, W, 3)
                """

                image = tf.image.resize(image, (C.crop_size, C.crop_size))
                image = tf.clip_by_value(image, 0, 255) / 255.0
                image = image * 2 - 1

                return image

        return _map

    def make_dataset(self, num_repeat, all_images, gen, path=''):
        """Utility function to build dataset

        Args:
            num_repeat (tf.int64): Number of times the dataset should be repeated
            all_images (list): List containing name of each target image
            gen (python class): Generator class w.r.t A or B
            path (str): Path to folder containing each target image, e.g. './datasets/monet2photo/trainA'

        Returns:
            tf.data.Dataset instance
        """

        gen.all_images = all_images

        dataset = tf.data.Dataset.from_generator(gen.data_generator, gen.dtype, tf.TensorShape([self.C.crop_size, self.C.crop_size, 3]))

        if self.shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer_size)
        
        dataset = dataset.map(self.map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(self.C.batch_size, drop_remainder=self.drop_remainder)

        dataset = dataset.repeat(num_repeat).prefetch(self.C.num_prefetch_batch)

        return dataset

    def make_zip_dataset(self):
        """Utility function to build zip dataset

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset, int] (A_dataset, B_dataset, dataset_length)
        """

        if self.training:
            a_path = join(self.path, 'trainA')
            b_path = join(self.path, 'trainB')
        else:
            a_path = join(self.path, 'testA')
            b_path = join(self.path, 'testB')

        A_all_images = list(paths.list_images(a_path))
        B_all_images = list(paths.list_images(b_path))

        if self.repeat:
            A_repeat = B_repeat = None
        else:
            if len(A_all_images) >= len(B_all_images):
                A_repeat = 1
                B_repeat = None
            else:
                A_repeat = None
                B_repeat = 1
        
        # build datasets
        A_dataset = self.make_dataset(A_repeat, A_all_images, self.gen_a, path=a_path)
        B_dataset = self.make_dataset(B_repeat, B_all_images, self.gen_b, path=b_path)

        dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
        dataset_length = max(len(A_all_images), len(B_all_images)) // self.C.batch_size


        return dataset, dataset_length