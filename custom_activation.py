# Some customized activation functions
import tensorflow as tf


def my_custom_activation(x, coeff):
    # Define your custom activation function here
    return tf.math.relu(x) * coeff


# Register the activation function for serialization
tf.keras.utils.register_keras_serializable()(my_custom_activation)


def my_custom_activation_get_config():
    # Define a get_config method for serialization if necessary
    return {}


# Register the custom activation function with its get_config method
tf.keras.utils.register_keras_serializable()(my_custom_activation_get_config)
