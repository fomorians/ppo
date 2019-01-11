import tensorflow as tf
import tensorflow.contrib.eager as tfe

from ppo import math


class Normalizer(tf.keras.Model):
    def __init__(self, shape, center=True, scale=True, clip=None):
        super(Normalizer, self).__init__()

        self.center = center
        self.scale = scale
        self.clip = clip

        self.count = tfe.Variable(0, dtype=tf.int32, trainable=False)
        self.mean = tfe.Variable(
            tf.zeros(shape=shape, dtype=tf.float32), trainable=False)
        self.var_sum = tfe.Variable(
            tf.zeros(shape=shape, dtype=tf.float32), trainable=False)

    @property
    def std(self):
        return tf.sqrt(
            tf.maximum(self.var_sum / tf.to_float(self.count - 1), 0))

    def call(self, inputs, weights, training=False):
        mask = tf.to_float(tf.not_equal(weights, 0))

        if training:
            self.count.assign_add(tf.to_int32(tf.reduce_sum(mask)))

            mean_deltas = tf.reduce_sum(
                (inputs - self.mean[None, None, ...]) * mask, axis=(0, 1))
            new_mean = self.mean + (mean_deltas / tf.to_float(self.count))

            var_deltas = (inputs - self.mean[None, None, ...]) * (
                inputs - new_mean[None, None, ...])
            new_var_sum = self.var_sum + tf.reduce_sum(
                var_deltas * mask, axis=(0, 1))

            self.mean.assign(new_mean)
            self.var_sum.assign(new_var_sum)

        if self.center:
            inputs -= self.mean[None, None, ...]
            inputs *= mask

        if self.scale:
            std = tf.where(
                math.is_near(self.std, 0.0), tf.ones_like(self.std), self.std)
            inputs /= std[None, None, ...]
            inputs *= mask

        if self.clip:
            inputs = tf.clip_by_value(inputs, -self.clip, self.clip)

        inputs = tf.check_numerics(inputs, 'inputs (post-normalization)')
        return inputs
