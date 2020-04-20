"""Core module w.r.t model related operations"""
import tensorflow as tf

from src import join

def save_models_weights(models_dict, output_dir):
    """Utility function to save many models weights

    Args:
        models (Dict{"model name": tf.keras.model}): Dict containing keras models
        output_dir (str): Folder to save the models
    """

    for model_name, model in models_dict.items():
        out_dir = join(output_dir, 'weights', model_name + '.h5')
        model.save_weights(out_dir)

def stop_training(models_dict):
    """Utility function to stop training on each given model

    Args:
        models (Dict{"model name": tf.keras.model}): Dict containing keras models
    """

    for model in models_dict:
        model.stop_training = True