import tensorflow_addons as tfa
import tensorflow as tf

def get_norm_layer(norm):
    """Utility function to get the normalization layer
    """

    if norm == None:
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return tf.keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return tf.keras.layers.LayerNormalization

def get_swish():

    def swish(x):
        """
        Swish activation function: x * sigmoid(x).
        Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        """
        return x * tf.keras.backend.sigmoid(x)

    return swish

def get_activation(activation):
    """Utility function to get the activation function
    """

    if activation == 'relu':
        return tf.nn.relu
    elif activation == 'swish':
        return get_swish

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'untruncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}