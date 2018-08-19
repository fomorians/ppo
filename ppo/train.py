import tensorflow as tf


def clip_gradients(grads, grad_clipping=None):
    if grad_clipping is not None:
        grads, _ = tf.clip_by_global_norm(grads, grad_clipping)
    return grads


def copy_variables(source_vars, dest_vars):
    assert len(source_vars) == len(dest_vars), 'vars must be the same length'
    for dest_var, source_var in zip(dest_vars, source_vars):
        dest_var.assign(source_var)
