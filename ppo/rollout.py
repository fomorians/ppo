import numpy as np
import tensorflow as tf


class BatchRollout:
    def __init__(self, env, max_episode_steps):
        self.env = env
        self.max_episode_steps = max_episode_steps

    def __call__(self, policy, episodes, render=False):
        assert len(self.env) == episodes

        observation_space = self.env.observation_space
        action_space = self.env.action_space

        observations = np.zeros(
            shape=(episodes, self.max_episode_steps) + observation_space.shape,
            dtype=observation_space.dtype,
        )
        actions = np.zeros(
            shape=(episodes, self.max_episode_steps) + action_space.shape,
            dtype=action_space.dtype,
        )
        rewards = np.zeros(shape=(episodes, self.max_episode_steps), dtype=np.float32)
        weights = np.zeros(shape=(episodes, self.max_episode_steps), dtype=np.float32)

        batch_size = len(self.env)
        episode_done = np.zeros(shape=batch_size, dtype=np.bool)

        observation = self.env.reset()

        for step in range(self.max_episode_steps):
            if render:
                self.env.envs[0].render()

            observation_tensor = tf.convert_to_tensor(observation, dtype=tf.float32)
            action_tensor = policy(observation_tensor[:, None, ...], training=False)
            action = action_tensor[:, 0].numpy()

            observation_next, reward, done, info = self.env.step(action)

            observations[:, step] = observation
            actions[:, step] = action
            rewards[:, step] = reward
            weights[:, step] = np.where(episode_done, 0.0, 1.0)

            # update episode done status
            episode_done = episode_done | done

            # end the rollout if all episodes are done
            if np.all(episode_done):
                break

            observation = observation_next

        # ensure rewards are masked
        rewards *= weights

        observations = tf.convert_to_tensor(observations)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        weights = tf.convert_to_tensor(weights)

        return observations, actions, rewards, weights
