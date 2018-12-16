import os
import gym
import atexit
import random
import argparse
import numpy as np
import tensorflow as tf

from tqdm import trange

from ppo import losses
from ppo.train import clip_gradients, copy_variables
from ppo.value import Value
from ppo.policy import Policy
from ppo.params import HyperParams
from ppo.rollout import Rollout
from ppo.targets import compute_advantages, compute_returns
from ppo.normalizer import Normalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True, help='Job directory')
    parser.add_argument('--render', action='store_true', help='Enable render')
    parser.add_argument('--seed', default=67, type=int, help='Random seed')
    parser.add_argument('--env', default='Pendulum-v0', help='Env name')
    args, _ = parser.parse_known_args()
    print('args:', vars(args))

    # make job dir
    if not os.path.exists(args.job_dir):
        os.makedirs(args.job_dir)

    # eager
    tf.enable_eager_execution()

    # environment
    env = gym.make(args.env)
    atexit.register(env.close)
    spec = env.unwrapped.spec

    # params
    discount_factor = 1 - (1 / spec.max_episode_steps)
    params = HyperParams(discount_factor=discount_factor)
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
    policy_old = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        scale=params.scale)
    value_fn = Value(observation_space=env.observation_space)

    # rewards
    rewards_normalizer = Normalizer(
        shape=(), center=False, scale=False, clip=None)

    # checkpoints
    checkpoint = tf.train.Checkpoint(
        global_step=global_step,
        optimizer=optimizer,
        policy=policy,
        value_fn=value_fn,
        rewards_normalizer=rewards_normalizer)

    # summaries
    summary_writer = tf.contrib.summary.create_file_writer(args.job_dir)
    summary_writer.set_as_default()

    # rollouts
    rollout = Rollout(env, spec.max_episode_steps)

    # priming
    # XXX: TF eager does not initialize weights until they're called
    rollout(policy=policy, episodes=1, training=False)
    rollout(policy=policy_old, episodes=1, training=False)

    # training iterations
    with trange(params.train_iters) as pbar:
        for it in pbar:
            # sample training transitions
            transitions = rollout(
                policy=policy, episodes=params.episodes, training=True)

            states = tf.convert_to_tensor(transitions.states)
            actions = tf.convert_to_tensor(transitions.actions)
            rewards = tf.convert_to_tensor(transitions.rewards)
            weights = tf.convert_to_tensor(transitions.weights)

            rewards_norm = rewards_normalizer(rewards, training=True)
            returns = compute_returns(
                rewards_norm,
                discount_factor=params.discount_factor,
                weights=weights)

            # targets
            values = value_fn(states, training=False)
            advantages = compute_advantages(
                rewards=rewards_norm,
                values=values,
                discount_factor=params.discount_factor,
                lambda_factor=params.lambda_factor,
                weights=weights,
                normalize=True)

            # update old policy
            copy_variables(
                source_vars=policy.trainable_variables,
                dest_vars=policy_old.trainable_variables)

            dist_old = policy_old.get_distribution(states, training=False)
            log_probs_old = dist_old.log_prob(actions)
            log_probs_old = tf.check_numerics(log_probs_old, 'log_probs_old')

            # training epochs
            for epoch in range(params.epochs):
                with tf.GradientTape() as tape:
                    dist = policy.get_distribution(states, training=False)
                    values = value_fn(states, training=False)

                    entropy = dist.entropy()
                    entropy = tf.check_numerics(entropy, 'entropy')

                    log_probs = dist.log_prob(actions)
                    log_probs = tf.check_numerics(log_probs, 'log_probs')

                    # losses
                    policy_loss = losses.policy_ratio_loss(
                        log_probs=log_probs,
                        log_probs_old=log_probs_old,
                        advantages=advantages,
                        epsilon_clipping=params.epsilon_clipping,
                        weights=weights)
                    value_loss = params.value_coef * (
                        tf.losses.mean_squared_error(
                            predictions=values,
                            labels=returns,
                            weights=weights))
                    entropy_loss = -params.entropy_coef * (
                        tf.losses.compute_weighted_loss(
                            losses=entropy, weights=weights))
                    loss = policy_loss + value_loss + entropy_loss

                # optimization
                trainable_variables = (
                    policy.trainable_variables + value_fn.trainable_variables)
                grads = tape.gradient(loss, trainable_variables)
                grads_clipped = clip_gradients(grads, params.grad_clipping)
                grads_and_vars = zip(grads_clipped, trainable_variables)
                optimizer.apply_gradients(
                    grads_and_vars, global_step=global_step)

                kl = tf.distributions.kl_divergence(dist, dist_old)
                entropy_mean = tf.losses.compute_weighted_loss(
                    losses=entropy, weights=weights)

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
                                              rewards_normalizer.mean)
                    tf.contrib.summary.scalar('rewards/train/std',
                                              rewards_normalizer.std)
                    tf.contrib.summary.scalar('rewards/train',
                                              transitions.episodic_reward)

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
            transitions = rollout(
                policy=policy,
                episodes=params.episodes,
                training=False,
                render=args.render)
            pbar.set_description('reward: {:.4f}'.format(
                transitions.episodic_reward))

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('rewards/eval',
                                          transitions.episodic_reward)

            # save checkpoint
            checkpoint_prefix = os.path.join(args.job_dir, 'ckpt')
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    main()
