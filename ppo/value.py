import pyoneer as pynr
import tensorflow as tf


class Value(tf.keras.Model):
    def __init__(self, observation_space):
        super(Value, self).__init__()

        self.observation_space = observation_space

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)

        self.dense1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)

        self.dense_value = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=kernel_initializer)

    def call(self, inputs, training=False):
        loc, var = pynr.nn.moments_from_range(self.observation_space.low,
                                              self.observation_space.high)
        inputs = pynr.math.normalize(inputs, loc=loc, scale=tf.sqrt(var))

        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)
        value = self.dense_value(hidden)

        return value[..., 0]
