import numpy as np
import tensorflow as tf


def swish(x):
    """
    Self-gating activation function.
    """
    y = x * tf.sigmoid(x)
    tf.check_numerics(y, 'swish')
    return y


def softplus_inv(x):
    """
    Inversion of softplus used for initialization.
    """
    y = tf.log(tf.exp(x) - 1)
    tf.check_numerics(y, 'softplus_inv')
    return y


def normalize(x, low, high):
    """
    Normalize to standard distribution.
    """
    mean = (high + low) / 2
    stddev = (high - low) / 2
    stddev = np.where(np.isclose(stddev, 0.0), 1.0, stddev)
    x = (x - mean) / stddev
    x = tf.check_numerics(x, 'normalize')
    return x


def denormalize(x, low, high):
    """
    Denormalize to original distribution.
    """
    mean = (high + low) / 2
    stddev = (high - low) / 2
    stddev = np.where(np.isclose(stddev, 0.0), 1.0, stddev)
    x = (x * stddev) + mean
    x = tf.check_numerics(x, 'denormalize')
    return x


def scale(x, low, high):
    """
    Scale from [min..max] to [-1..1].
    """
    x = (x - low) / (high - low)
    x = (x * 2) + 1
    x = tf.check_numerics(x, 'scale')
    return x


def unscale(x, low, high):
    """
    Unscale from [-1..1] to [min..max].
    """
    x = (x + 1) / 2
    x = x * (high - low) + low
    x = tf.check_numerics(x, 'unscale')
    return x


def is_near(x, y, rtol=1e-05, atol=1e-08):
    """
    Check if x is near y.
    """
    tol = atol + rtol * tf.abs(y)
    diff = tf.abs(x - y)
    condition = tf.reduce_all(tf.less(diff, tol))
    return condition
