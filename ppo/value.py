import pyoneer as pynr
import tensorflow as tf


class Value(tf.keras.Model):
    def __init__(self, observation_space, **kwargs):
        super(Value, self).__init__(**kwargs)

        self.observation_space = observation_space

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

        self.dense1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.activations.swish,
            kernel_initializer=kernel_initializer,
        )
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.activations.swish,
            kernel_initializer=kernel_initializer,
        )

        self.dense_value = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=kernel_initializer
        )

    def call(self, inputs, training=False):
        loc, var = pynr.moments.range_moments(
            self.observation_space.low, self.observation_space.high
        )
        inputs = pynr.math.normalize(inputs, loc=loc, scale=tf.sqrt(var))

        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)
        value = self.dense_value(hidden)

        return value[..., 0]
