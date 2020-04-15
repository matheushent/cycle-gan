"""Core module w.r.t summary writing"""
import tensorflow as tf

def summary(data_dict, step=None, name='summary'):

    """Utility function to write logs in summary

    Args:
        data_dict (dict): Dictionary containing logs
        step (int): Related epoch
    """

    def _summary(name, data):
        tf.summary.scalar(name, data, step=step)
    
    with tf.name_scope(name):
        for name, data in data_dict.items():
            _summary(name, data)