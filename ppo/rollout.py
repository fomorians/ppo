import attr
import numpy as np

from ppo.transitions import Transitions


@attr.s
class Rollout(object):
    env = attr.ib()
    max_episode_steps = attr.ib()

    def __call__(self, policy, episodes, training=False, render=False):
        states = np.zeros(
            shape=(episodes, self.max_episode_steps), dtype=np.int32)
        actions = np.zeros(
            shape=(episodes, self.max_episode_steps), dtype=np.int32)
        rewards = np.zeros(
            shape=(episodes, self.max_episode_steps), dtype=np.float32)
        weights = np.zeros(
            shape=(episodes, self.max_episode_steps), dtype=np.float32)

        for episode in range(episodes):
            state = self.env.reset()

            for step in range(self.max_episode_steps):
                if render:
                    self.env.render()

                state_batch = np.reshape(state, (1, 1))
                action_batch = policy(
                    state_batch, training=training, return_numpy=True)
                action = action_batch[0, 0]

                next_state, reward, done, info = self.env.step(action)

                states[episode, step] = state
                actions[episode, step] = action
                rewards[episode, step] = reward
                weights[episode, step] = 1.0

                if done:
                    break

                state = next_state

        return Transitions(states, actions, rewards, weights)
