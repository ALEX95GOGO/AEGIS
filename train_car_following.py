"""
Training script for the car-following task.

Notes for maintainers:
- This file is formatted for readability and GitHub readiness (PEP8-ish).
- Function signatures and overall behavior are unchanged.
- Assumes external modules (env_multiagent_dis, human_model, algo.TD3_car_following) are correct.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from typing import Tuple

import numpy as np
import pygame
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from env_multiagent_dis import CarFollowing
from human_model import Actor as h_Model
from algo.TD3_car_following import DRL  # noqa: N813 (keep original class name usage)

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------


def signal_handler(sig, frame) -> None:
    """Gracefully terminate on Ctrl+C (SIGINT)."""
    print("Procedure terminated!")
    try:
        pygame.display.quit()
        pygame.quit()
    finally:
        sys.exit(0)


def set_seed(seed: int) -> None:
    """Set random seeds and deterministic flags for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

DIRECTORY = "./ddpg_save/"

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--bc", action="store_true", default=False)
parser.add_argument(
    "--algorithm",
    type=int,
    default=0,
    help="RL algorithm (0: Proposed, 1: IARL, 2: HIRL, 3: Vanilla TD3)",
)
parser.add_argument(
    "--human_model",
    action="store_true",
    default=False,
    help="Whether to use the human behavior model",
)
parser.add_argument(
    "--human_model_update",
    action="store_true",
    default=False,
    help="Whether to update the human behavior model",
)
parser.add_argument(
    "--maximum_episode",
    type=float,
    default=500,
    help="Maximum training episode number",
)
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--initial_exploration_rate",
    type=float,
    default=0.5,
    help="Initial exploration policy variance",
)
parser.add_argument(
    "--cutoff_exploration_rate",
    type=float,
    default=0.05,
    help="Minimum exploration policy variance",
)
parser.add_argument(
    "--exploration_decay_rate",
    type=float,
    default=0.99988,
    help="Decay factor of exploration policy variance",
)
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Resume trained agents from checkpoint",
)
parser.add_argument(
    "--warmup",
    action="store_true",
    default=False,
    help="Start training only after collecting enough data",
)
parser.add_argument(
    "--warmup_threshold",
    type=int,
    default=1e4,
    help="Warmup length by step",
)
parser.add_argument(
    "--reward_shaping",
    type=int,
    default=1,
    help="Reward shaping scheme (0: none; 1: proposed)",
)
parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
parser.add_argument(
    "--simulator_port",
    type=int,
    default=2000,
    help="CARLA port (for multiple clients)",
)
parser.add_argument(
    "--simulator_render_frequency",
    type=int,
    default=40,
    help="CARLA rendering frequency (Hz)",
)
parser.add_argument(
    "--simulator_conservative_surrounding",
    action="store_true",
    default=False,
    help="Whether surrounding vehicles are conservative",
)
parser.add_argument(
    "--joystick_enabled",
    action="store_true",
    default=False,
    help="Use Logitech G29 joystick for human guidance",
)

parser.add_argument("--tau", default=0.005, type=float, help="Target smoothing coeff")
parser.add_argument("--target_update_interval", default=10, type=int)
parser.add_argument("--test_iteration", default=10, type=int)

parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--gamma", default=0.99, type=int, help="Discount factor")
parser.add_argument("--capacity", default=1_000_000, type=int, help="Replay size")
parser.add_argument("--batch_size", default=100, type=int, help="Minibatch size")

parser.add_argument("--sample_frequency", default=2000, type=int)
parser.add_argument("--render", default=False, type=bool, help="Show UI or not")
parser.add_argument("--log_interval", default=50, type=int)
parser.add_argument("--load", default=False, type=bool, help="Load model")
parser.add_argument(
    "--render_interval",
    default=100,
    type=int,
    help="After this interval, env.render() will work",
)
parser.add_argument("--exploration_noise", default=0.1, type=float)
parser.add_argument("--max_episode", default=100000, type=int, help="Num of games")
parser.add_argument("--print_log", default=5, type=int)
parser.add_argument("--update_iteration", default=10, type=int)
parser.add_argument("--idm", action="store_true", default=False)
parser.add_argument("--eye", action="store_true", default=False)

args = parser.parse_args()

# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------


def train_car_following_task() -> None:
    """Main training loop for the car-following task."""
    set_seed(args.seed)

    # TensorBoard
    with SummaryWriter("./algo/checkpoints/log") as writer:
        log_dir = "algo/checkpoints/human_att_epoch_3.tar"

        # Environment
        env = CarFollowing(
            joystick_enabled=args.joystick_enabled,
            conservative_surrounding=args.simulator_conservative_surrounding,
            frame=args.simulator_render_frequency,
            port=args.simulator_port,
        )

        s_dim = [env.observation_size_width, env.observation_size_height]
        a_dim = env.action_size

        # Agent
        DRL_agent = DRL(a_dim, s_dim, device=args.device)  # keep variable name pattern

        exploration_rate = args.initial_exploration_rate

        # Resume
        if args.resume and os.path.exists(log_dir):
            checkpoint = torch.load(log_dir)
            DRL_agent.load(log_dir)
            start_epoch = checkpoint["epoch"] + 1
        else:
            start_epoch = 0

        # (Optional) behavior cloning warm start
        if args.bc:
            h_model = h_Model().cuda()
            if os.path.isfile(args.bc):
                print(f"=> loading checkpoint 'algo/checkpoints/human_att_epoch_3.tar'")
                checkpoint = torch.load("algo/checkpoints/human_att_epoch_3.tar")
                start_epoch = 1
                h_model.load_state_dict(checkpoint["state_dict"])
                DRL_agent.actor.load_state_dict(
                    torch.load("algo/checkpoints/human_att_epoch_3.tar")["state_dict"]
                )
                print(f"=> loaded checkpoint '{args.bc}' (epoch {checkpoint['epoch']})")
            else:
                print(f"=> no checkpoint found at '{args.bc}'")

        # Global trackers
        total_step = 0
        a_loss = c_loss = dis_loss = kl_loss = 0
        q_target = cross_correlation = kl_ours = nss = mae = td_errors = sim = 0

        loss_critic_hist, loss_actor_hist = [], []
        episode_total_reward_list, episode_mean_reward_list = [], []
        global_reward_list, episode_duration_list = [], []

        # per-episode logs
        max_ep = int(args.maximum_episode)
        drl_action = [[] for _ in range(max_ep)]
        adopted_action = [[] for _ in range(max_ep)]
        reward_i_record = [[] for _ in range(max_ep)]
        reward_e_record = [[] for _ in range(max_ep)]

        start_time = time.perf_counter()

        # Training episodes
        for i in range(start_epoch, max_ep):
            # Episode init
            ep_reward = 0.0
            step = 0
            done = False

            # Reset env
            observation = env.reset()
            state = np.repeat(np.expand_dims(observation, 2), 3, axis=2)
            state_ = state.copy()

            while not done:
                # DRL acts
                action, dis = DRL_agent.choose_action(state)
                dis = dis.cpu().numpy()
                action = np.clip(
                    np.random.normal(action, exploration_rate), -1.0, 1.0
                )
                drl_action[i].append(action)

                # Environment step
                (
                    observation_,
                    action_fdbk,
                    reward_e,
                    _,
                    done,
                    scope,
                    dis_front,
                    travel_dis,
                    ttc,
                    rgb,
                ) = env.step(action)

                # Shift state window
                state_1 = state_[:, :, 2].copy()
                state_[:, :, 0] = state_[:, :, 1].copy()
                state_[:, :, 1] = state_1
                state_[:, :, 2] = observation_.copy()

                reward = reward_e  # (shaping hook if needed)

                # If previous metrics were tensors, detach for logging
                if isinstance(cross_correlation, torch.Tensor):
                    cross_correlation = cross_correlation.detach().cpu().numpy().item()
                    kl_ours = kl_ours.detach().cpu().numpy().item()

                # Learn (after warmup)
                learn_threshold = args.warmup_threshold if args.warmup else 256
                if total_step > learn_threshold:
                    (
                        c_loss,
                        a_loss,
                        dis_loss,
                        kl_loss,
                        q_target,
                        cross_correlation,
                        kl_ours,
                        nss,
                        mae,
                        td_errors,
                        sim,
                    ) = DRL_agent.learn(batch_size=16, epoch=i, eye=args.eye)

                    q_target = q_target.cpu()
                    cross_correlation = cross_correlation.cpu()
                    kl_ours = kl_ours.cpu()
                    mae = mae.cpu()
                    loss_critic_hist.append(np.average(c_loss))
                    loss_actor_hist.append(np.average(a_loss))

                    # Exploration anneal
                    if exploration_rate > args.cutoff_exploration_rate:
                        exploration_rate *= args.exploration_decay_rate
                    else:
                        exploration_rate = args.cutoff_exploration_rate

                # Episode accounting
                ep_reward += reward
                global_reward_list.append([reward_e])
                reward_e_record[i].append(reward_e)
                adopted_action[i].append(action)

                observation = observation_.copy()
                state = state_.copy()

                dura = env.terminate_position
                total_step += 1
                step += 1

                # Handle Ctrl+C
                signal.signal(signal.SIGINT, signal_handler)
            mean_reward = ep_reward / max(step, 1)
            episode_total_reward_list.append(ep_reward)
            episode_mean_reward_list.append(mean_reward)
            episode_duration_list.append(dura)

            print(f"\n episode: {i}")
            print(f" explore_rate: {round(exploration_rate, 4)}")
            print(f" c_loss: {round(np.average(c_loss), 4)}")
            print(f" a_loss: {round(np.average(a_loss), 4)}")
            print(f" total_step: {total_step}")
            print(f" episode_step: {step}")
            print(f" episode_cumu_reward: {round(ep_reward, 4)}")
            print(f" episode_mean_reward: {round(mean_reward, 4)}")

            writer.add_scalar("reward/reward_episode", ep_reward, i)
            writer.add_scalar(
                "reward/reward_episode_noshaping", np.mean(reward_e_record[i]), i
            )
            writer.add_scalar("reward/duration_episode", step, i)
            writer.add_scalar("reward/survival_distance", travel_dis, i)
            writer.add_scalar("rate_exploration", round(exploration_rate, 4), i)
            writer.add_scalar("loss/loss_critic", round(np.average(c_loss), 4), i)
            writer.add_scalar("loss/loss_actor", round(np.average(a_loss), 4), i)
            writer.add_scalar("loss/loss_distance", dis_loss, i)
            writer.add_scalar("loss/loss_kl", kl_loss, i)
            writer.add_scalar("loss/loss_sim", sim, i)
            writer.add_scalar("loss/q_target", np.average(q_target), i)
            writer.add_scalar("loss/td_errors", np.average(td_errors), i)
            writer.add_scalar("loss/cross_correlation", np.average(cross_correlation), i)
            writer.add_scalar("loss/kl_ours", np.average(kl_ours), i)
            writer.add_scalar("loss/nss", np.average(nss), i)
            writer.add_scalar("loss/mae", np.average(mae), i)

        # Post-training
        print(f"total time: {time.perf_counter() - start_time:.2f}s")

        timeend = round(time.time())
        DRL_agent.save_actor("./algo/models_leftturn", timeend)

        pygame.display.quit()
        pygame.quit()

        # Save episode logs
        action_drl = drl_action[0:i]
        action_final = adopted_action[0:i]
        scio.savemat(
            f"dataleftturn_{args.seed}-{args.algorithm}-{timeend}.mat",
            mdict={
                "action_drl": action_drl,
                "action_final": action_final,
                "stepreward": global_reward_list,
                "mreward": episode_mean_reward_list,
                "step": episode_duration_list,
                "reward": episode_total_reward_list,
                "r_i": reward_i_record,
                "r_e": reward_e_record,
            },
        )


# -------------------------------------------------------------------------
# Entry
# -------------------------------------------------------------------------

if __name__ == "__main__":
    train_car_following_task()
