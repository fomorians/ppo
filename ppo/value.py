import tensorflow as tf

from ppo import math


class Value(tf.keras.Model):
    def __init__(self, observation_space):
        super(Value, self).__init__()

        self._observation_space = observation_space

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)

        self.dense1 = tf.keras.layers.Dense(
            units=128,
            activation=math.swish,
            kernel_initializer=kernel_initializer)

        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=math.swish,
            kernel_initializer=kernel_initializer)

        self.dense_value = tf.keras.layers.Dense(
            units=1, activation=None, kernel_initializer=kernel_initializer)

    def call(self, inputs, training=False):
        inputs = math.normalize(inputs, self._observation_space.low,
                                self._observation_space.high)

        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)

        value = self.dense_value(hidden)

        value = tf.squeeze(value, axis=-1)
        value = tf.check_numerics(value, 'value')
        return value
