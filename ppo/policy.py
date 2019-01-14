import pyoneer as pynr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


class Policy(tf.keras.Model):
    def __init__(self, observation_space, action_space, scale):
        super(Policy, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        action_size = self.action_space.shape[0]

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        scale_initializer = pynr.initializers.SoftplusInverse(scale=scale)

        self.dense1 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=pynr.nn.swish,
            kernel_initializer=kernel_initializer)
        self.dense_loc = tf.keras.layers.Dense(
            units=action_size,
            activation=tf.tanh,
            kernel_initializer=kernel_initializer)
        self.scale_diag_inverse = tfe.Variable(
            scale_initializer([scale] * action_size), trainable=True)

    @property
    def scale_diag(self):
        return tf.nn.softplus(self.scale_diag_inverse)

    def call(self, inputs, training=None):
        loc, var = pynr.nn.moments_from_range(self.observation_space.low,
                                              self.observation_space.high)
        inputs = pynr.math.normalize(inputs, loc=loc, scale=tf.sqrt(var))

        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)

        loc = pynr.math.rescale(
            self.dense_loc(hidden),
            oldmin=-1.0,
            oldmax=1.0,
            newmin=self.action_space.low,
            newmax=self.action_space.high)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=self.scale_diag)
        return dist
