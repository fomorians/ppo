import os
import gym
import atexit
import random
import argparse
import numpy as np
import pyoneer as pynr
import pyoneer.rl as pyrl
import tensorflow as tf

from tqdm import trange

from ppo.value import Value
from ppo.policy import Policy
from ppo.params import HyperParams
from ppo.rollout import Rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True, help='Job directory')
    parser.add_argument('--render', action='store_true', help='Enable render')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--env', default='Pendulum-v0', help='Env name')
    args, _ = parser.parse_known_args()
    print('args:', args)

    # make job dir
    if not os.path.exists(args.job_dir):
        os.makedirs(args.job_dir)

    # eager
    tf.enable_eager_execution()

    # environment
    env = gym.make(args.env)
    atexit.register(env.close)

    # params
    params = HyperParams()
    params_path = os.path.join(args.job_dir, 'params.json')
    params.save(params_path)
    print('params:', params)

    # seeding
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # optimization
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

    # models
    policy = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        scale=params.scale)
    policy_anchor = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        scale=params.scale)
    baseline = Value(observation_space=env.observation_space)

    # strategies
    exploration_strategy = pyrl.strategies.SampleStrategy(policy)
    inference_strategy = pyrl.strategies.ModeStrategy(policy)

    # rewards
    rewards_moments = pynr.nn.ExponentialMovingMoments(shape=(), rate=0.9)

    # checkpoints
    checkpoint = tf.train.Checkpoint(
        global_step=global_step,
        optimizer=optimizer,
        policy=policy,
        baseline=baseline,
        rewards_moments=rewards_moments)

    # summaries
    summary_writer = tf.contrib.summary.create_file_writer(args.job_dir)
    summary_writer.set_as_default()

    # rollouts
    rollout = Rollout(env, env.spec.max_episode_steps)

    # priming
    # NOTE: TF eager does not initialize weights until they're called
    anchor_inference_strategy = pyrl.strategies.ModeStrategy(policy_anchor)
    rollout(policy=inference_strategy, episodes=1)
    rollout(policy=anchor_inference_strategy, episodes=1)

    # training iterations
    with trange(params.train_iters) as pbar:
        for it in pbar:
            # sample training transitions
            states, actions, rewards, weights = rollout(
                policy=exploration_strategy, episodes=params.episodes)

            rewards_moments(rewards, weights=weights, training=True)
            rewards_norm = pynr.math.normalize(
                rewards,
                loc=rewards_moments.mean,
                scale=rewards_moments.std,
                weights=weights)

            returns = pyrl.targets.discounted_rewards(
                rewards_norm,
                discount_factor=params.discount_factor,
                weights=weights)

            # targets
            values = baseline(states, training=False)
            advantages = pyrl.targets.generalized_advantages(
                rewards=rewards_norm,
                values=values,
                discount_factor=params.discount_factor,
                lambda_factor=params.lambda_factor,
                weights=weights,
                normalize=True)

            # update old policy
            pynr.training.update_target_variables(
                source_variables=policy.trainable_variables,
                target_variables=policy_anchor.trainable_variables)

            policy_dist_anchor = policy_anchor(states, training=False)
            log_probs_anchor = policy_dist_anchor.log_prob(actions)
            log_probs_anchor = tf.check_numerics(log_probs_anchor,
                                                 'log_probs_anchor')

            # training epochs
            for epoch in range(params.epochs):
                with tf.GradientTape() as tape:
                    policy_dist = policy(states, training=True)
                    values = baseline(states, training=True)

                    entropy = policy_dist.entropy()
                    entropy = tf.check_numerics(entropy, 'entropy')

                    log_probs = policy_dist.log_prob(actions)
                    log_probs = tf.check_numerics(log_probs, 'log_probs')

                    # losses
                    policy_loss = pyrl.losses.clipped_policy_gradient_loss(
                        log_probs=log_probs,
                        log_probs_anchor=log_probs_anchor,
                        advantages=advantages,
                        epsilon_clipping=params.epsilon_clipping,
                        weights=weights)
                    value_loss = tf.losses.mean_squared_error(
                        predictions=values,
                        labels=returns,
                        weights=weights * params.value_coef)
                    entropy_loss = -tf.losses.compute_weighted_loss(
                        losses=entropy, weights=weights * params.entropy_coef)
                    loss = policy_loss + value_loss + entropy_loss

                # optimization
                trainable_variables = (
                    policy.trainable_variables + baseline.trainable_variables)
                grads = tape.gradient(loss, trainable_variables)
                if params.grad_clipping is not None:
                    grads_clipped, _ = tf.clip_by_global_norm(
                        grads, params.grad_clipping)
                grads_and_vars = zip(grads_clipped, trainable_variables)
                optimizer.apply_gradients(
                    grads_and_vars, global_step=global_step)

                kl = tf.distributions.kl_divergence(policy_dist,
                                                    policy_dist_anchor)
                entropy_mean = tf.losses.compute_weighted_loss(
                    losses=entropy, weights=weights)
                episodic_reward = tf.reduce_mean(
                    tf.reduce_sum(rewards, axis=-1))

                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('gradient_norm/unclipped',
                                              tf.global_norm(grads))
                    tf.contrib.summary.scalar('gradient_norm',
                                              tf.global_norm(grads_clipped))

                    tf.contrib.summary.scalar('losses/entropy', entropy_loss)
                    tf.contrib.summary.scalar('losses/policy', policy_loss)
                    tf.contrib.summary.scalar('losses/value', value_loss)
                    tf.contrib.summary.scalar('losses/loss', loss)

                    tf.contrib.summary.scalar('rewards/train/mean',
                                              rewards_moments.mean)
                    tf.contrib.summary.scalar('rewards/train/std',
                                              rewards_moments.std)
                    tf.contrib.summary.scalar('rewards/train', episodic_reward)

                    tf.contrib.summary.scalar('entropy', entropy_mean)
                    tf.contrib.summary.scalar('kl', kl)

                    tf.contrib.summary.histogram('states', states)
                    tf.contrib.summary.histogram('actions', actions)
                    tf.contrib.summary.histogram('rewards', rewards)
                    tf.contrib.summary.histogram('rewards/norm', rewards_norm)
                    tf.contrib.summary.histogram('advantages', advantages)
                    tf.contrib.summary.histogram('returns', returns)
                    tf.contrib.summary.histogram('values', values)

            # evaluation
            if it % params.eval_interval == 0:
                states, actions, rewards, weights = rollout(
                    policy=inference_strategy,
                    episodes=params.episodes,
                    render=args.render)
                episodic_reward = tf.reduce_mean(
                    tf.reduce_sum(rewards, axis=-1))
                pbar.set_description('reward: {:.4f}'.format(
                    episodic_reward.numpy()))

                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('rewards/eval', episodic_reward)

            # save checkpoint
            checkpoint_prefix = os.path.join(args.job_dir, 'ckpt')
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    main()
