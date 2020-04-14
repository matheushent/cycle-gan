import tensorflow as tf
from src.path import *
import json
import pickle

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')

    if len(gpus) > 0:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        return '/GPU:0'
    else:
        return '/CPU:0'

def save_json(path, obj):

    with open(path, 'w') as f:
        json.dump(obj, f)

def load_json(path):

    with open(path, 'r') as f:
        return json.load(f)

def save_yaml(path, data):
    import oyaml as yaml

    with open(path, 'w') as f:
        yaml.dump(data, f)

def load_yaml(path):
    import oyaml as yaml

    with open(path, 'r') as f:
        return yaml.load(f)

def save_pickle(path, obj):

    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):

    with open(path, 'rb') as f:
        return pickle.load(f)