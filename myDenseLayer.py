# A customized dense layer. Serialization and deserialization is needed for saving and loading models
import tensorflow as tf


class myDense(tf.keras.layers.Layer):
    def __init__(self, out_features, activation, kernel_initializer=None, bias_initializer=None, layer_name=None, **kwargs):
        super().__init__(**kwargs)
        tf.random.set_seed(73)
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer or tf.keras.initializers.GlorotNormal(
            seed=73)
        self.bias_initializer = bias_initializer or tf.keras.initializers.Zeros()
        self.layer_name = layer_name

    def build(self, input_shape):  # Create the state of the layer (weights)
        self.w = self.add_weight(shape=(input_shape[-1], self.out_features),
                                 initializer=self.kernel_initializer,
                                 name='w')
        self.b = self.add_weight(shape=(self.out_features,),
                                 initializer=self.bias_initializer,
                                 name='b')
        self.coeff_activation = self.add_weight(shape=(),
                                                initializer=tf.constant_initializer(
                                                    1.0),
                                                name='coeff_act')

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b, self.coeff_activation)

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_features": self.out_features,
            "activation": tf.keras.activations.serialize(self.activation),
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "layer_name": self.layer_name
        })
        return config

    @classmethod
    def from_config(cls, config):
        activation = tf.keras.activations.deserialize(config.pop("activation"))
        config["activation"] = activation
        config["kernel_initializer"] = tf.keras.initializers.deserialize(
            config.pop("kernel_initializer"))
        config["bias_initializer"] = tf.keras.initializers.deserialize(
            config.pop("bias_initializer"))
        return cls(**config)
