import attr
import numpy as np
import tensorflow as tf

from ppo.transitions import Transitions


@attr.s
class Rollout(object):
    env = attr.ib()
    max_episode_steps = attr.ib()

    def __call__(self, policy, episodes, training=False, render=False):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]

        states = np.zeros(
            shape=(episodes, self.max_episode_steps, state_size),
            dtype=np.float32)
        actions = np.zeros(
            shape=(episodes, self.max_episode_steps, action_size),
            dtype=np.float32)
        rewards = np.zeros(
            shape=(episodes, self.max_episode_steps), dtype=np.float32)
        weights = np.zeros(
            shape=(episodes, self.max_episode_steps), dtype=np.float32)

        for episode in range(episodes):
            state = self.env.reset()

            for step in range(self.max_episode_steps):
                if render:
                    self.env.render()

                state_batch = tf.convert_to_tensor(
                    state[None, None, ...], dtype=np.float32)
                action_batch = policy(state_batch, training=training)
                action = action_batch[0, 0].numpy()

                next_state, reward, done, info = self.env.step(action)

                states[episode, step] = state
                actions[episode, step] = action
                rewards[episode, step] = reward
                weights[episode, step] = 1.0

                if done:
                    break

                state = next_state

        return Transitions(states, actions, rewards, weights)
