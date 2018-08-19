import tensorflow as tf

# from ppo import rnn
from ppo import math


class Value(tf.keras.Model):
    def __init__(self, observation_space):
        super(Value, self).__init__()

        self._observation_space = observation_space

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)

        # self.rnn = rnn.RNN(num_units=100)

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
        inputs = tf.one_hot(inputs, self._observation_space.n)

        # hidden = self.rnn(inputs, training=training)

        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)

        value = self.dense_value(hidden)
        """
        V(s_t) = sum([gamma**i for i in range(t, T)]) * V_1(s_t) + V_2(s_t)
        """
        # value1 = dense_value[..., 0]
        # value2 = dense_value[..., 1]
        # value = discounts * value1 + value2

        value = tf.squeeze(value, axis=-1)
        value = tf.check_numerics(value, 'value')
        return value
