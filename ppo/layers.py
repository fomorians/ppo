import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils


class StatelessGaussianDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(StatelessGaussianDropout, self).__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate

    def call(self, inputs, training=None, seed=None):
        def noised():
            stddev = np.sqrt(self.rate / (1.0 - self.rate))
            if seed is not None:
                noise = tf.contrib.stateless.stateless_random_normal(
                    shape=inputs.shape, seed=(seed, seed + 1))
            else:
                noise = K.random_normal(shape=inputs.shape)
            noised = inputs * (1.0 + noise * stddev)
            return K.in_train_phase(noised, inputs, training=training)

        return tf.cond(0 < self.rate < 1, noised, lambda: inputs)

    def get_config(self):
        config = {'rate': self.rate.numpy()}
        base_config = super(StatelessGaussianDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
