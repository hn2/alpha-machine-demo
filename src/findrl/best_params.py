#   https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py

from typing import Any, Callable, Dict, Union

import numpy as np
import optuna
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def best_ppo_params():
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """

    activation_fn = 'tanh'
    batch_size = 128
    clip_range = 0.1
    ent_coef = 0.00505853
    gae_lambda = 0.9
    gamma = 0.99
    learning_rate = 1e-05
    max_grad_norm = 0.9
    n_epochs = 10
    n_steps = 2048
    net_arch = 'verybig'
    use_sde = True
    vf_coef = 0.367562
    ortho_init = False

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
        "verybig": [dict(pi=[512, 512, 512, 512], vf=[512, 512, 512, 512])],
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_best_freq": sde_best_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


def best_a2c_params():
    """
    Sampler for A2C hyperparams.

    :param trial:
    :return:
    """
    activation_fn = 'tanh'
    ent_coef = 0.00225096
    gae_lambda = 0.95
    gamma = 0.995
    learning_rate = 1e-05
    lr_schedule = 'linear'
    max_grad_norm = 0.3
    n_steps = 64
    net_arch = 'verybig'
    normalize_advantage = True
    ortho_init = True
    use_rms_prop = False
    use_sde = False
    vf_coef = 0.0253355

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
        "verybig": [dict(pi=[512, 512, 512, 512], vf=[512, 512, 512, 512])],
    }[net_arch]

    # sde_net_arch = {
    #     None: None,
    #     "tiny": [64],
    #     "small": [64, 64],
    # }[sde_net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "normalize_advantage": normalize_advantage,
        "max_grad_norm": max_grad_norm,
        "use_rms_prop": use_rms_prop,
        "vf_coef": vf_coef,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            # full_std=full_std,
            activation_fn=activation_fn,
            # sde_net_arch=sde_net_arch,
            ortho_init=ortho_init,
        ),
    }


def best_sac_params():
    """
    Sampler for SAC hyperparams.

    :param trial:
    :return:
    """

    # batch_size = 128
    # buffer_size = 100000
    # gamma = 0.995
    # learning_rate = 1e-05
    # learning_starts = 20000
    # log_std_init = 0.438489
    # net_arch = 'verybig'
    # tau = 0.005
    # train_freq = 4
    # use_sde = True
    # use_sde_at_warmup = True

    #   batch_size = 64
    batch_size = 512
    buffer_size = 1000000
    gamma = 0.999
    learning_rate = 1e-05
    learning_starts = 10000
    log_std_init = -2.7557
    net_arch = 'verybig'
    tau = 0.08
    train_freq = 4
    # use_sde = False
    use_sde = True
    use_sde_at_warmup = True

    gradient_steps = train_freq
    ent_coef = "auto"

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        "verybig": [512, 512, 512, 512],
        # Uncomment for tuning HER
        # "large": [256, 256, 256],
        # "verybig": [512, 512, 512],
    }[net_arch]

    target_entropy = "auto"
    # if ent_coef == 'auto':
    #     # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
    #     target_entropy = trial.suggest_uniform('target_entropy', -10, 10)

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "use_sde": use_sde,
        "use_sde_at_warmup": use_sde_at_warmup,
        "target_entropy": target_entropy,
        "policy_kwargs": dict(log_std_init=log_std_init, net_arch=net_arch),
    }

    # if trial.using_her_replay_buffer:
    #     hyperparams = best_her_params(trial, hyperparams)

    return hyperparams


def best_td3_params():
    """
    Sampler for TD3 hyperparams.

    :param trial:
    :return:
    """
    # batch_size = 2048
    # batch_size = 512
    batch_size = 512
    buffer_size = 10000
    gamma = 0.9
    learning_rate = 1e-05
    net_arch = 'verybig'
    tau = 0.001
    train_freq = 64
    gradient_steps = train_freq

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        "verybig": [512, 512, 512, 512],
        # Uncomment for tuning HER
        # "verybig": [256, 256, 256],
    }[net_arch]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(net_arch=net_arch),
        "tau": tau,
    }

    # if noise_type == "normal":
    #     hyperparams["action_noise"] = NormalActionNoise(
    #         mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
    #     )
    # elif noise_type == "ornstein-uhlenbeck":
    #     hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
    #         mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
    #     )

    # if trial.using_her_replay_buffer:
    #     hyperparams = best_her_params(trial, hyperparams)

    return hyperparams


def best_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DDPG hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    gradient_steps = train_freq

    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_uniform("noise_std", 0, 1)

    # NOTE: Add "verybig" to net_arch when tuning HER (see TD3)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
    }[net_arch]

    hyperparams = {
        "gamma": gamma,
        "tau": tau,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )

    # if trial.using_her_replay_buffer:
    #     hyperparams = best_her_params(trial, hyperparams)

    return hyperparams


def best_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DQN hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)])
    exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 5000, 10000, 20000])

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
    subbest_steps = trial.suggest_categorical("subbest_steps", [1, 2, 4, 8])
    gradient_steps = max(train_freq // subbest_steps, 1)

    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])

    net_arch = {"tiny": [64], "small": [64, 64], "medium": [256, 256]}[net_arch]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    # if trial.using_her_replay_buffer:
    #     hyperparams = best_her_params(trial, hyperparams)

    return hyperparams


def best_her_params(trial: optuna.Trial, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sampler for HerReplayBuffer hyperparams.

    :param trial:
    :parma hyperparams:
    :return:
    """
    her_kwargs = trial.her_kwargs.copy()
    her_kwargs["n_sampled_goal"] = trial.suggest_int("n_sampled_goal", 1, 5)
    her_kwargs["goal_selection_strategy"] = trial.suggest_categorical(
        "goal_selection_strategy", ["final", "episode", "future"]
    )
    her_kwargs["online_sampling"] = trial.suggest_categorical("online_sampling", [True, False])
    hyperparams["replay_buffer_kwargs"] = her_kwargs
    return hyperparams


def best_tqc_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TQC hyperparams.

    :param trial:
    :return:
    """
    # TQC is SAC + Distributional RL
    hyperparams = best_sac_params(trial)

    n_quantiles = trial.suggest_int("n_quantiles", 5, 50)
    top_quantiles_to_drop_per_net = trial.suggest_int("top_quantiles_to_drop_per_net", 0, n_quantiles - 1)

    hyperparams["policy_kwargs"].update({"n_quantiles": n_quantiles})
    hyperparams["top_quantiles_to_drop_per_net"] = top_quantiles_to_drop_per_net

    return hyperparams


def best_qrdqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for QR-DQN hyperparams.

    :param trial:
    :return:
    """
    # TQC is DQN + Distributional RL
    hyperparams = best_dqn_params(trial)

    n_quantiles = trial.suggest_int("n_quantiles", 5, 200)
    hyperparams["policy_kwargs"].update({"n_quantiles": n_quantiles})

    return hyperparams


HYPERPARAMS = {
    "a2c": best_a2c_params,
    "ddpg": best_ddpg_params,
    "dqn": best_dqn_params,
    "qrdqn": best_qrdqn_params,
    "sac": best_sac_params,
    "tqc": best_tqc_params,
    "ppo": best_ppo_params,
    "td3": best_td3_params,
}
