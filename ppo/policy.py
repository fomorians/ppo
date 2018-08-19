import tensorflow as tf

# from ppo import rnn
from ppo import math
from ppo import layers


class Policy(tf.keras.Model):
    def __init__(self, observation_space, action_space, rate):
        super(Policy, self).__init__()

        self._observation_space = observation_space
        self._action_space = action_space

        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)

        # self.rnn = rnn.RNN(num_units=100)

        self.dense1 = tf.keras.layers.Dense(
            units=128,
            activation=math.swish,
            kernel_initializer=kernel_initializer)
        self.drop_dense1 = layers.StatelessGaussianDropout(rate=rate)

        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=math.swish,
            kernel_initializer=kernel_initializer)
        self.drop_dense2 = layers.StatelessGaussianDropout(rate=rate)

        self.dense_logits = tf.keras.layers.Dense(
            units=self._action_space.n,
            activation=None,
            kernel_initializer=kernel_initializer)

    def get_distribution(self, inputs, training=False, seed=None):
        inputs = tf.one_hot(inputs, self._observation_space.n)

        # hidden = self.rnn(inputs, training=training)

        hidden = self.drop_dense1(
            self.dense1(inputs), training=training, seed=seed)
        hidden = self.drop_dense2(
            self.dense2(hidden), training=training, seed=seed)
        logits = self.dense_logits(hidden)

        dist = tf.distributions.Categorical(logits=logits)
        return dist

    def call(self, inputs, training=False, seed=None, return_numpy=False):
        dist = self.get_distribution(inputs, training=training, seed=seed)

        if training:
            action = dist.sample()
        else:
            action = dist.mode()

        if return_numpy:
            action = action.numpy()

        return action
