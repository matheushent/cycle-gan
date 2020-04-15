from __future__ import division

from warnings import filterwarnings
filterwarnings('ignore')

from optparse import OptionParser
import matplotlib.pyplot as plt
import pickle
import json
import time

import numpy as np

from src.architectures.conv_disc import ConvDiscriminator
from src.architectures.resnet_gen import ResNetGenerator
from src.model import save_models_weights, stop_training
from src.callbacks.calls import LinearDecay, Callbacks
from src import join, mkdir, check_gpu, save_pickle
from src.data.make_data import MakeDataset
from src.config import Struct, Config
from src.summary import summary

from tensorflow.keras import utils
import tensorflow as tf

parser = OptionParser()

parser.add_option("-d", dest="dataset", help="Dataset to use", default="monet2photo")
parser.add_option("--dd", dest="dataset_dir", help="Path to datasets", default="datasets")
parser.add_options("-e", type="int", dest="epochs", help="Epochs to run training", default=200)
parser.add_options("--ed", type="int", dest="epoch_decay", help="Epoch to start decaying learning rate", default=100)
parser.add_option("--si", type="int", dest="sample_interval", help="Interval to make samples", default=20)
parser.add_option("-p", type="int", dest="patience", help="int indicating how many epochs without loss improvement is need to break training", default=10)
parser.add_option("--mn", dest="model_name", help="Model name to identify specific model")

(options, args) = parser.parse_args()

if not options.model_name:
    parser.error("You must pass --mn argument")

# check if GPU is available. If is, use it automatically
device = check_gpu()

# instantiate config class containing some params
C = Config()

C.model_name = options.model_name
C.dataset = options.dataset

# output directory
output_dir = join('output', options.dataset, options.model_name)
mkdir(output_dir)

# save configurations
save_pickle(join(output_dir, 'config.pickle'), C)

# build datasets
make_train_data = MakeDataset(C, join(options.dataset, options.dataset_dir), training=True, repeat=False)
train_dataset, dataset_length = make_train_data.make_zip_dataset()

make_test_data = MakeDataset(C, join(options.dataset, options.dataset_dir), training=False, shuffle=False, repeat=True)
test_dataset = make_test_data.make_zip_dataset()

# build linear decay
linear_decay = LinearDecay(C.learning_rate, options.epochs, options.epoch_decay, C.beta_1)

# define input shape
input_shape = (C.crop_size, C.crop_size, 3)

# build discriminators
disc_a = ConvDiscriminator(
    input_shape=input_shape,
    dim=64, num_downsamplings=3, norm='instance_norm',
    lr_scheduler=linear_decay)
disc_b = ConvDiscriminator(
    input_shape=input_shape,
    dim=64, num_downsamplings=3, norm='instance_norm',
    lr_scheduler=linear_decay)

# build generators
gen_A2B = ResNetGenerator(
    input_shape=input_shape,
    output_channels=3, dim=64,
    num_downsamplings=2, num_blocks=9,
    norm='instance_norm', activation='swish')
gen_B2A = ResNetGenerator(
    input_shape=input_shape,
    output_channels=3, dim=64,
    num_downsamplings=2, num_blocks=9,
    norm='instance_norm', activation='swish')

# input images from both domains
img_A = tf.keras.Input(shape=input_shape)
img_B = tf.keras.Input(shape=input_shape)

# translate images to the other domain
fake_B = gen_A2B.model(img_A)
fake_A = gen_B2A.model(img_B)

# translate images back to original domain
reconstructed_A = gen_B2A.model(fake_B)
reconstructed_B = gen_A2B.model(fake_A)

# identity mapping of images
img_A_id = gen_B2A.model(img_A)
img_B_id = gen_A2B.model(img_B)

# we'll combine models but only train the generators
disc_a.model.trainable = False
disc_b.model.trainable = False

# discriminators determine validity of translated images
valid_A = disc_a.model(fake_A)
valid_B = disc_b.model(fake_B)

combined_model = tf.keras.Model(
    inputs=[img_A, img_B],
    outputs=[
        valid_A, valid_B,
        reconstructed_A, reconstructed_B,
        img_A_id, img_B_id]
    )
combined_model.compile(
    loss=[
        'mse', 'mse',
        'mae', 'mae',
        'mae', 'mae'],
    loss_weights=[
        1, 1,
        C.cycle_loss_weight, C.cycle_loss_weight,
        C.identity_loss_weight, C.identity_loss_weight],
        optimizer=disc_a.optimizer
)

utils.plot_model(combined_model, join(output_dir, 'combined_model.png'))

# adversarial loss ground truths
disc_patch = (C.crop_size / 2**4, C.crop_size / 2**4, 1) # output shape of discriminator
valid = tf.ones((C.batch_size,) + disc_patch)
fake = tf.zeros((C.batch_size,) + disc_patch)

# sample
test_iter = iter(test_dataset)
sample_dir = join(output_dir, 'samples_training')
mkdir(sample_dir)

# params
epoch_length = 100
iter_num = 0
best_loss = np.Inf
last_epoch = 0

# summary
summary_writer = tf.summary.create_file_writer(join(output_dir, 'summary', 'train'))

# start training
start_time = time.time()

# train on GPU if available
with tf.device(device):

    for epoch in range(options.epochs):

        progbar = utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch + 1, options.epochs))

        for idx, (A, B) in enumerate(train_dataset):

            ## train discriminators

            # translate images to opposite domain
            fake_B = gen_A2B.model.predict(A)
            fake_A = gen_B2A.model.predict(B)

            # train discriminators (original image = real / translated = fake)
            A_real_loss = disc_a.model.train_on_batch(A, valid)
            A_fake_loss = disc_a.model.train_on_batch(fake_A, fake)
            A_loss = 0.5 * tf.math.add(A_real_loss, A_fake_loss)

            B_real_loss = disc_b.model.train_on_batch(B, valid)
            B_fake_loss = disc_b.model.train_on_batch(fake_B, fake)
            B_loss = 0.5 * tf.math.add(B_real_loss, B_fake_loss)

            # total discriminator loss
            discriminator_loss = 0.5 * tf.math.add(A_loss, B_loss)

            ## train generators

            generator_loss = combined_model.train_on_batch(
                [A, B],
                [valid, valid,
                ]
            )

            elapsed_time = time.time() - start_time

            # update progbar
            progbar.update(iter_num + 1, [
                ('D loss', discriminator_loss[0]),
                ('D acc', 100*discriminator_loss[1]),
                ('G loss', generator_loss[0]),
                ('G adv', np.mean(generator_loss[1:3])),
                ('G recon', np.mean(generator_loss[3:5])),
                ('G id', np.mean(generator_loss[5:6])),
                ('Elapsed time', elapsed_time)
            ])

            if disc_a.optimizer.iterations.numpy() % options.sample_interval == 0:
                _A, _B = next(test_iter)

                # translate images to other domain
                _fake_B = gen_A2B.model.predict(_A)
                _fake_A = gen_B2A.model.predict(_B)

                # translate back to original domain
                _reconstructed_A = gen_B2A.model.predict(_fake_B)
                _reconstructed_B = gen_A2B.model.predict(_fake_A)

                generated_images = np.concatenate([
                    _A, _fake_B, _reconstructed_A,
                    _B, _fake_A, _reconstructed_B
                ])

                # rescale images
                generated_images *= 0.5
                generated_images += 0.5

                fig, ax = plt.subplots(C.r, C.c)
                cnt = 0
                for i in range(C.r):
                    for j in range(C.c):
                        ax[i, j].imshow(generated_images[cnt])
                        ax[i, j].set_title(C.titles[j])
                        ax[i, j].axis('off')
                fig.savefig(join(sample_dir, "sample_{}_{}.png".format(epoch, idx)))
                plt.close()

        summary_dict = {
            "D loss": discriminator_loss[0],
            "D acc": 100*discriminator_loss[1],
            "G loss": generator_loss[0],
            "G adv": np.mean(generator_loss[1:3]),
            "G recon": np.mean(generator_loss[3:5]),
            "G id": np.mean(generator_loss[5:6]),
            "Learning rate": disc_a.lr_scheduler.current_learning_rate
        }

        # write scalars in summary
        with summary_writer.as_default():
            summary(summary_dict, step=epoch)

        # saving models in case the loss has improved
        if generator_loss < best_loss:

            print("Loss improved from {} to {}".format(best_loss, generator_loss))
            print("Saving models in {}".format(output_dir))
            models_dict = {
                "gen_A2B": gen_A2B.model,
                "gen_B2A": gen_B2A.model,
                "disc_a": disc_a.model,
                "disc_b": disc_b.model,
                "combined_model": combined_model
            }
            save_models_weights(models_dict, output_dir)

            # early stopping
            if epoch - last_epoch > options.patience:
                print("Loss hasn't improved in {} and {} epochs. Stop training now".format(last_epoch, epoch))
                stop_training(models_dict)

            last_epoch = epoch

print("Training finished. Exitting")