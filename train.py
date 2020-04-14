from __future__ import division

from warnings import filterwarnings
filterwarnings('ignore')

from optparse import OptionParser
import pickle
import json

from src import join, mkdir, check_gpu, save_pickle
from src.config import Struct, Config
from src.data.make_data import MakeDataset

from tensorflow.keras import utils
import tensorflow as tf

parser = OptionParser()

parser.add_option("-d", dest="dataset", help="Dataset to use", default="monet2photo")
parser.add_option("--dd", dest="dataset_dir", help="Path to datasets", default="datasets")
parser.add_options("-e", type="int", dest="epochs", help="Epochs to run training", default=200)
parser.add_options("--ed", type="int", dest="epoch_decay", help="Epoch to start decaying learning rate", default=100)
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