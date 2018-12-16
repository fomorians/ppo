import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp

from ppo import math


class Policy(tf.keras.Model):
    def __init__(self, observation_space, action_space, scale):
        super(Policy, self).__init__()

        self._observation_space = observation_space
        self._action_space = action_space

        action_size = self._action_space.shape[0]

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)

        self.dense1 = tf.keras.layers.Dense(
            units=128,
            activation=math.swish,
            kernel_initializer=kernel_initializer)
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=math.swish,
            kernel_initializer=kernel_initializer)
        self.dense_loc = tf.keras.layers.Dense(
            units=action_size,
            activation=None,
            kernel_initializer=kernel_initializer)
        self.scale_diag_inverse = tfe.Variable(
            math.softplus_inv([scale] * action_size), trainable=True)

    @property
    def scale_diag(self):
        return tf.nn.softplus(self.scale_diag_inverse)

    def get_distribution(self, inputs, training=None):
        inputs = math.normalize(inputs, self._observation_space.low,
                                self._observation_space.high)

        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)
        loc = self.dense_loc(hidden)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=self.scale_diag)
        return dist

    def call(self, inputs, training=None):
        dist = self.get_distribution(inputs, training=training)

        if training:
            action = dist.sample()
        else:
            action = dist.mode()

        return action
