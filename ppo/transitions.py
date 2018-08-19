import attr
import tensorflow as tf


@attr.s
class Transitions(object):
    states = attr.ib()
    actions = attr.ib()
    rewards = attr.ib()
    weights = attr.ib()

    @property
    def episodic_reward(self):
        return tf.reduce_mean(tf.reduce_sum(self.rewards, axis=-1)).numpy()
