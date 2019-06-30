import os
import gym
import atexit
import random
import argparse
import numpy as np
import pyoneer as pynr
import pyoneer.rl as pyrl
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm import trange
from tensorflow.python.keras.utils import losses_utils

from ppo.models import Value, Policy
from ppo.params import HyperParams
from ppo.rollout import BatchRollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True, help="Job directory")
    parser.add_argument(
        "--render", action="store_true", help="Enable evaluation render"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--env", default="Pendulum-v0", help="Env name")
    args, _ = parser.parse_known_args()
    print("args:", args)

    # make job dir
    os.makedirs(args.job_dir, exist_ok=True)

    # params
    params = HyperParams()
    params_path = os.path.join(args.job_dir, "params.json")
    params.save(params_path)
    print("params:", params)

    # environment
    env = pyrl.wrappers.Batch(
        lambda batch_id: gym.make(args.env), batch_size=params.episodes
    )
    atexit.register(env.close)

    # seeding
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # optimization
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)

    # models
    policy = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        scale=params.scale,
    )
    baseline = Value(observation_space=env.observation_space)

    # strategies
    exploration_strategy = pyrl.strategies.Sample(policy)
    inference_strategy = pyrl.strategies.Mode(policy)

    # normalization
    rewards_moments = pynr.moments.ExponentialMovingMoments(
        shape=(), rate=params.reward_decay
    )

    # checkpoints
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        policy=policy,
        baseline=baseline,
        rewards_moments=rewards_moments,
    )
    checkpoint_path = tf.train.latest_checkpoint(args.job_dir)
    if checkpoint_path is not None:
        checkpoint.restore(checkpoint_path)

    # summaries
    summary_writer = tf.summary.create_file_writer(
        args.job_dir, max_queue=100, flush_millis=5 * 60 * 1000
    )
    summary_writer.set_as_default()

    # rollouts
    rollout = BatchRollout(env, max_episode_steps=env.spec.max_episode_steps)

    # prime models
    # NOTE: TF eager does not initialize weights until they're called
    mock_states = tf.zeros(
        shape=(1, 1, env.observation_space.shape[0]), dtype=np.float32
    )
    policy(mock_states, training=False)

    advantages_fn = pyrl.targets.GeneralizedAdvantages(
        discount_factor=params.discount_factor,
        lambda_factor=params.lambda_factor,
        normalize=True,
    )
    returns_fn = pyrl.targets.DiscountedReturns(discount_factor=params.discount_factor)

    value_loss_fn = tf.losses.MeanSquaredError()
    policy_loss_fn = pyrl.losses.ClippedPolicyGradient(
        epsilon_clipping=params.epsilon_clipping
    )
    entropy_loss_fn = pyrl.losses.PolicyEntropy()

    # training iterations
    with trange(params.train_iters) as pbar:
        for it in pbar:
            # sample training transitions
            states, actions, rewards, weights = rollout(
                policy=exploration_strategy, episodes=params.episodes
            )
            episodic_reward = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))

            rewards_moments(rewards, sample_weight=weights, training=True)
            rewards_norm = pynr.math.normalize(
                rewards,
                loc=rewards_moments.mean,
                scale=rewards_moments.std,
                sample_weight=weights,
            )

            values = baseline(states, training=False)

            # targets
            advantages = advantages_fn(
                rewards=rewards_norm, values=values, sample_weight=weights
            )
            returns = returns_fn(rewards=rewards_norm, sample_weight=weights)

            policy_anchor_dist = policy(states, training=False)
            log_probs_anchor = policy_anchor_dist.log_prob(actions)

            tf.summary.scalar(
                "rewards/train/mean", rewards_moments.mean, step=optimizer.iterations
            )
            tf.summary.scalar(
                "rewards/train/std", rewards_moments.std, step=optimizer.iterations
            )
            tf.summary.scalar(
                "rewards/train", episodic_reward, step=optimizer.iterations
            )

            tf.summary.histogram("states", states, step=optimizer.iterations)
            tf.summary.histogram("actions", actions, step=optimizer.iterations)
            tf.summary.histogram("rewards", rewards, step=optimizer.iterations)
            tf.summary.histogram(
                "rewards/norm", rewards_norm, step=optimizer.iterations
            )
            tf.summary.histogram("advantages", advantages, step=optimizer.iterations)
            tf.summary.histogram("returns", returns, step=optimizer.iterations)
            tf.summary.histogram("values", values, step=optimizer.iterations)

            # training epochs
            for epoch in range(params.epochs):
                with tf.GradientTape() as tape:
                    # forward passes
                    policy_dist = policy(states, training=True)
                    values = baseline(states, training=True)

                    entropy = policy_dist.entropy()
                    log_probs = policy_dist.log_prob(actions)

                    # losses
                    policy_loss = policy_loss_fn(
                        log_probs=log_probs,
                        log_probs_anchor=log_probs_anchor,
                        advantages=advantages,
                        sample_weight=weights,
                    )
                    value_loss = value_loss_fn(
                        y_pred=values[..., None],
                        y_true=returns[..., None],
                        sample_weight=weights[..., None] * params.value_coef,
                    )
                    entropy_loss = entropy_loss_fn(
                        entropy=entropy, sample_weight=weights * params.entropy_coef
                    )
                    loss = policy_loss + value_loss + entropy_loss

                # optimization
                trainable_variables = (
                    policy.trainable_variables + baseline.trainable_variables
                )
                grads = tape.gradient(loss, trainable_variables)
                if params.grad_clipping is not None:
                    grads, _ = tf.clip_by_global_norm(grads, params.grad_clipping)
                grads_and_vars = zip(grads, trainable_variables)
                optimizer.apply_gradients(grads_and_vars)

                # summaries
                kl = tf.reduce_mean(
                    tfp.distributions.kl_divergence(policy_dist, policy_anchor_dist)
                )
                entropy_mean = losses_utils.compute_weighted_loss(
                    losses=entropy, sample_weight=weights
                )
                gradient_norm = tf.linalg.global_norm(grads)

                tf.summary.scalar(
                    "losses/entropy", entropy_loss, step=optimizer.iterations
                )
                tf.summary.scalar(
                    "losses/policy", policy_loss, step=optimizer.iterations
                )
                tf.summary.scalar("losses/value", value_loss, step=optimizer.iterations)
                tf.summary.scalar("losses/loss", loss, step=optimizer.iterations)
                tf.summary.scalar("entropy", entropy_mean, step=optimizer.iterations)
                tf.summary.scalar("kl", kl, step=optimizer.iterations)
                tf.summary.scalar(
                    "gradient_norm", gradient_norm, step=optimizer.iterations
                )

            # evaluation
            if it % params.eval_interval == params.eval_interval - 1:
                states, actions, rewards, weights = rollout(
                    policy=inference_strategy,
                    episodes=params.episodes,
                    render=args.render,
                )
                episodic_reward = tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
                pbar.set_description("reward: {:.4f}".format(episodic_reward.numpy()))

                tf.summary.scalar(
                    "rewards/eval", episodic_reward, step=optimizer.iterations
                )

            # save checkpoint
            checkpoint_prefix = os.path.join(args.job_dir, "ckpt")
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == "__main__":
    main()
