#   https://stable-baselines3.readthedocs.io/en/master/modules/her.html#example
#   https://github.com/takuseno/d3rlpy
import uuid
from datetime import datetime, timedelta
from typing import Callable

import d3rlpy as d3
import numpy as np
import pandas as pd
import stable_baselines3 as sb3
import torch
# from d3rlpy.algos import TD3, SAC
# from d3rlpy.algos import DDPG, DQN, SAC, TD3
# from stable_baselines3 import HerReplayBuffer, A2C, PPO, DDPG, DQN, SAC, TD3
from stable_baselines3.a2c.policies import MlpPolicy as A2cMlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.ddpg.policies import MlpPolicy as DdpgMlpPolicy
from stable_baselines3.dqn.policies import MlpPolicy as DqnMlpPolicy
from stable_baselines3.ppo.policies import MlpPolicy as PpoMlpPolicy
from stable_baselines3.sac.policies import MlpPolicy as SacMlpPolicy
from stable_baselines3.td3.policies import MlpPolicy as Td3MlpPolicy


# from sb3_contrib import ARS, MaskablePPO, RecurrentPPO, QRDQN, TQC, TRPO

def get_model_name(delimeter, model_prefix, number_of_instruments, total_timesteps, train_look_back_period, offset,
                   env_look_back_period, spread, market, resolution, online_algorithm, compute_position,
                   compute_indicators, compute_reward, meta_rl):
    v_uuid = (uuid.uuid4().hex)[:8]

    v_model_name = model_prefix + str(number_of_instruments) + delimeter + str(total_timesteps) + delimeter + str(
        train_look_back_period) + '_' + str(offset) + delimeter + str(env_look_back_period) + delimeter + \
                   str(int(
                       spread)) + delimeter + market + delimeter + resolution + delimeter + 'on_algo.' + online_algorithm.lower() + \
                   delimeter + 'c_pos.' + str(compute_position) + delimeter + 'c_ind.' + str(
        compute_indicators) + delimeter + \
                   'c_rew.' + str(compute_reward).replace("'", "") + delimeter + 'm_rl.' + str(
        meta_rl) + delimeter + str(v_uuid)

    return v_model_name


def parse_model_name(model_name, delimiter):
    v_model_prefix, v_number_of_instruments, v_total_timesteps, v_train_lookback_period, v_env_lookback_period, v_spread, v_market_name, v_resolution_name, v_online_algorithm_name, v_compute_position_name, v_compute_indicators_name, v_compute_reward_name, v_meta_rl, v_uuid = model_name.split(
        delimiter)

    v_number_of_instruments = int(v_number_of_instruments)
    v_env_lookback_period = int(v_env_lookback_period)

    v_online_algorithm = v_online_algorithm_name.split('.')[-1]
    v_compute_position = v_compute_position_name.split('.')[-1]
    v_compute_indicators = v_compute_indicators_name.split('.')[-1]
    v_compute_reward = v_compute_reward_name.split('.')[-1]
    v_meta_rl = v_meta_rl.split('.')[-1]

    return v_number_of_instruments, v_total_timesteps, v_market_name, v_resolution_name, v_env_lookback_period, v_spread, v_online_algorithm, v_compute_position, v_compute_indicators, v_compute_reward, v_meta_rl


def parse_model_name_full(model_name, delimiter):
    v_model_prefix, v_number_of_instruments, v_total_timesteps, v_train_lookback_period, v_env_lookback_period, v_spread, v_market_name, v_resolution_name, v_online_algorithm_name, v_compute_position_name, v_compute_indicators_name, v_compute_reward_name, v_meta_rl, v_uuid = model_name.split(
        delimiter)

    v_number_of_instruments = int(v_number_of_instruments)
    v_env_lookback_period = int(v_env_lookback_period)

    v_online_algorithm = v_online_algorithm_name.split('.')[-1]
    v_compute_position = v_compute_position_name.split('.')[-1]
    v_compute_indicators = v_compute_indicators_name.split('.')[-1]
    v_compute_reward = v_compute_reward_name.split('.')[-1]

    return v_model_prefix, v_number_of_instruments, v_total_timesteps, v_train_lookback_period, v_env_lookback_period, v_spread, v_market_name, v_resolution_name, v_online_algorithm, v_compute_position, v_compute_indicators, v_compute_reward, v_meta_rl


def parse_noise_from_prefix(prefix):
    idx = prefix.find('noise')

    return prefix[idx + 6:]


def parse_model_name_online_algorithm(model_name, delimiter):
    v_model_prefix, v_number_of_instruments, v_total_timesteps, v_train_lookback_period, v_env_lookback_period, v_spread, v_market_name, v_resolution_name, v_online_algorithm_name, v_compute_position_name, v_compute_indicators_name, v_compute_reward_name, v_meta_rl, v_uuid = model_name.split(
        delimiter)

    v_online_algorithm = v_online_algorithm_name.split('.')[-1]

    return v_online_algorithm


def parse_model_name_train_lookback_period(model_name, delimiter):
    v_model_prefix, v_number_of_instruments, v_total_timesteps, v_train_lookback_period, v_env_lookback_period, v_spread, v_market_name, v_resolution_name, v_online_algorithm_name, v_compute_position_name, v_compute_indicators_name, v_compute_reward_name, v_meta_rl, v_uuid = model_name.split(
        delimiter)

    return v_train_lookback_period.split('_')[0]


def parse_model_name_online_algorithm_row(row, delimiter):
    v_model_prefix, v_number_of_instruments, v_total_timesteps, v_train_lookback_period, v_env_lookback_period, v_spread, v_market_name, v_resolution_name, v_online_algorithm_name, v_compute_position_name, v_compute_indicators_name, v_compute_reward_name, v_meta_rl, v_uuid = \
        row['Model Name'].split(delimiter)

    v_online_algorithm = v_online_algorithm_name.split('.')[-1]

    return v_online_algorithm


def get_online_class_and_policy_sb3(algo):
    if algo.upper() == 'A2C':
        v_class = sb3.A2C
        v_policy = A2cMlpPolicy
    elif algo.upper() == 'PPO':
        v_class = sb3.PPO
        v_policy = PpoMlpPolicy
    elif algo.upper() == 'DDPG':
        v_class = sb3.DDPG
        v_policy = DdpgMlpPolicy
    elif algo.upper() == 'DQN':
        v_class = sb3.DQN
        v_policy = DqnMlpPolicy
    elif algo.upper() == 'SAC':
        v_class = sb3.SAC
        v_policy = SacMlpPolicy
    elif algo.upper() == 'TD3':
        v_class = sb3.TD3
        v_policy = Td3MlpPolicy

    return v_class, v_policy


def get_online_class_sb3(algo):
    if algo.upper() == 'A2C':
        v_class = sb3.A2C
    elif algo.upper() == 'PPO':
        v_class = sb3.PPO
    elif algo.upper() == 'DDPG':
        v_class = sb3.DDPG
    elif algo.upper() == 'DQN':
        v_class = sb3.DQN
    elif algo.upper() == 'SAC':
        v_class = sb3.SAC
    elif algo.upper() == 'TD3':
        v_class = sb3.TD3

    return v_class


# def get_online_class_and_policy_sb3_contrib(algo):
#     if algo.upper() == 'ARS':
#         v_class = sb3c.ARS
#         v_policy = A2cMlpPolicy
#     elif algo.upper() == 'PPO':
#         v_class = sb3.PPO
#         v_policy = PpoMlpPolicy
#     elif algo.upper() == 'DDPG':
#         v_class = sb3.DDPG
#         v_policy = DdpgMlpPolicy
#     elif algo.upper() == 'DQN':
#         v_class = sb3.DQN
#         v_policy = DqnMlpPolicy
#     elif algo.upper() == 'SAC':
#         v_class = sb3.SAC
#         v_policy = SacMlpPolicy
#     elif algo.upper() == 'TD3':
#         v_class = sb3.TD3
#         v_policy = Td3MlpPolicy
#
#     return v_class, v_policy
#
#
# def get_online_class_sb3_contrib(algo):
#     if algo.upper() == 'ARS':
#         v_class = sb3.A2C
#     elif algo.upper() == 'PPO':
#         v_class = sb3.PPO
#     elif algo.upper() == 'DDPG':
#         v_class = sb3.DDPG
#     elif algo.upper() == 'DQN':
#         v_class = sb3.DQN
#     elif algo.upper() == 'SAC':
#         v_class = sb3.SAC
#     elif algo.upper() == 'TD3':
#         v_class = sb3.TD3
#
#     return v_class


'''
online only:

d3rlpy.algos.BC
d3rlpy.algos.DDPG
d3rlpy.algos.TD3
d3rlpy.algos.SAC

online + offline:

d3rlpy.algos.BCQ
d3rlpy.algos.BEAR
d3rlpy.algos.CQL
d3rlpy.algos.AWAC
d3rlpy.algos.CRR
d3rlpy.algos.PLAS
d3rlpy.algos.TD3PlusBC
d3rlpy.algos.IQL

d3rlpy.algos.DDPG
d3rlpy.algos.TD3
d3rlpy.algos.SAC
d3rlpy.algos.BCQ
d3rlpy.algos.BEAR
d3rlpy.algos.CRR
d3rlpy.algos.CQL
d3rlpy.algos.AWR
d3rlpy.algos.AWAC
d3rlpy.algos.PLAS
d3rlpy.algos.PLASWithPerturbation
d3rlpy.algos.TD3PlusBC
d3rlpy.algos.IQL
d3rlpy.algos.MOPO
d3rlpy.algos.COMBO
d3rlpy.algos.RandomPolicy
'''


def get_class_d3rlpy(algo,
                     actor_learning_rate,
                     critic_learning_rate,
                     actor_encoder_factory,
                     critic_encoder_factory,
                     batch_size):
    if algo.upper() == 'DDPG':
        v_class = d3.algos.DDPG(actor_learning_rate=actor_learning_rate,
                                critic_learning_rate=critic_learning_rate,
                                actor_encoder_factory=actor_encoder_factory,
                                critic_encoder_factory=critic_encoder_factory,
                                batch_size=batch_size,
                                use_gpu=torch.cuda.is_available())
    elif algo.upper() == 'TD3':
        v_class = d3.algos.TD3(actor_learning_rate=actor_learning_rate,
                               critic_learning_rate=critic_learning_rate,
                               actor_encoder_factory=actor_encoder_factory,
                               critic_encoder_factory=critic_encoder_factory,
                               batch_size=batch_size,
                               use_gpu=torch.cuda.is_available())
    elif algo.upper() == 'SAC':
        v_class = d3.algos.SAC(actor_learning_rate=actor_learning_rate,
                               critic_learning_rate=critic_learning_rate,
                               actor_encoder_factory=actor_encoder_factory,
                               critic_encoder_factory=critic_encoder_factory,
                               batch_size=batch_size,
                               use_gpu=torch.cuda.is_available())
    elif algo.upper() == 'BCQ':
        v_class = d3.algos.BCQ(actor_learning_rate=actor_learning_rate,
                               critic_learning_rate=critic_learning_rate,
                               actor_encoder_factory=actor_encoder_factory,
                               critic_encoder_factory=critic_encoder_factory,
                               batch_size=batch_size,
                               use_gpu=torch.cuda.is_available())
    elif algo.upper() == 'BEAR':
        v_class = d3.algos.BEAR(actor_learning_rate=actor_learning_rate,
                                critic_learning_rate=critic_learning_rate,
                                actor_encoder_factory=actor_encoder_factory,
                                critic_encoder_factory=critic_encoder_factory,
                                batch_size=batch_size,
                                use_gpu=torch.cuda.is_available())
    elif algo.upper() == 'CRR':
        v_class = d3.algos.CRR(actor_learning_rate=actor_learning_rate,
                               critic_learning_rate=critic_learning_rate,
                               actor_encoder_factory=actor_encoder_factory,
                               critic_encoder_factory=critic_encoder_factory,
                               batch_size=batch_size,
                               use_gpu=torch.cuda.is_available())
    elif algo.upper() == 'CQL':
        v_class = d3.algos.CQL(actor_learning_rate=actor_learning_rate,
                               critic_learning_rate=critic_learning_rate,
                               actor_encoder_factory=actor_encoder_factory,
                               critic_encoder_factory=critic_encoder_factory,
                               batch_size=batch_size,
                               use_gpu=torch.cuda.is_available())
    elif algo.upper() == 'AWR':
        v_class = d3.algos.AWR(actor_learning_rate=actor_learning_rate,
                               critic_learning_rate=critic_learning_rate,
                               actor_encoder_factory=actor_encoder_factory,
                               critic_encoder_factory=critic_encoder_factory,
                               batch_size=batch_size,
                               use_gpu=torch.cuda.is_available())
    elif algo.upper() == 'AWAC':
        v_class = d3.algos.AWAC(actor_learning_rate=actor_learning_rate,
                                critic_learning_rate=critic_learning_rate,
                                actor_encoder_factory=actor_encoder_factory,
                                critic_encoder_factory=critic_encoder_factory,
                                batch_size=batch_size,
                                use_gpu=torch.cuda.is_available())
    elif algo.upper() == 'PLAS':
        v_class = d3.algos.PLAS(actor_learning_rate=actor_learning_rate,
                                critic_learning_rate=critic_learning_rate,
                                actor_encoder_factory=actor_encoder_factory,
                                critic_encoder_factory=critic_encoder_factory,
                                batch_size=batch_size,
                                use_gpu=torch.cuda.is_available())
    elif algo.upper() == 'PLASWITHPERTURBATION':
        v_class = d3.algos.PLASWithPerturbation(actor_learning_rate=actor_learning_rate,
                                                critic_learning_rate=critic_learning_rate,
                                                actor_encoder_factory=actor_encoder_factory,
                                                critic_encoder_factory=critic_encoder_factory,
                                                batch_size=batch_size,
                                                use_gpu=torch.cuda.is_available())
    elif algo.upper() == 'TD3PLUSBC':
        v_class = d3.algos.TD3PlusBC(actor_learning_rate=actor_learning_rate,
                                     critic_learning_rate=critic_learning_rate,
                                     actor_encoder_factory=actor_encoder_factory,
                                     critic_encoder_factory=critic_encoder_factory,
                                     batch_size=batch_size,
                                     use_gpu=torch.cuda.is_available())
    elif algo.upper() == 'IQL':
        v_class = d3.algos.IQL(actor_learning_rate=actor_learning_rate,
                               critic_learning_rate=critic_learning_rate,
                               actor_encoder_factory=actor_encoder_factory,
                               critic_encoder_factory=critic_encoder_factory,
                               batch_size=batch_size,
                               use_gpu=torch.cuda.is_available())
    elif algo.upper() == 'MOPO':
        v_class = d3.algos.MOPO(actor_learning_rate=actor_learning_rate,
                                critic_learning_rate=critic_learning_rate,
                                actor_encoder_factory=actor_encoder_factory,
                                critic_encoder_factory=critic_encoder_factory,
                                batch_size=batch_size,
                                use_gpu=torch.cuda.is_available())
    elif algo.upper() == 'COMBO':
        v_class = d3.algos.COMBO(actor_learning_rate=actor_learning_rate,
                                 critic_learning_rate=critic_learning_rate,
                                 actor_encoder_factory=actor_encoder_factory,
                                 critic_encoder_factory=critic_encoder_factory,
                                 batch_size=batch_size,
                                 use_gpu=torch.cuda.is_available())

    return v_class


def load_class_from_json_d3rlpy(algo,
                                model_json_file_name):
    if algo.upper() == 'DDPG':
        v_model = d3.algos.DDPG.from_json(model_json_file_name)
    elif algo.upper() == 'TD3':
        v_model = d3.algos.TD3.from_json(model_json_file_name)
    elif algo.upper() == 'SAC':
        v_model = d3.algos.SAC.from_json(model_json_file_name)
    elif algo.upper() == 'BCQ':
        v_model = d3.algos.BCQ.from_json(model_json_file_name)
    elif algo.upper() == 'BEAR':
        v_model = d3.algos.BEAR.from_json(model_json_file_name)
    elif algo.upper() == 'CRR':
        v_model = d3.algos.CRR.from_json(model_json_file_name)
    elif algo.upper() == 'CQL':
        v_model = d3.algos.CQL.from_json(model_json_file_name)
    elif algo.upper() == 'AWR':
        v_model = d3.algos.AWR.from_json(model_json_file_name)
    elif algo.upper() == 'AWAC':
        v_model = d3.algos.AWAC.from_json(model_json_file_name)
    elif algo.upper() == 'PLAS':
        v_model = d3.algos.PLAS.from_json(model_json_file_name)
    elif algo.upper() == 'PLASWithPerturbation':
        v_model = d3.algos.PLASWithPerturbation.from_json(model_json_file_name)
    elif algo.upper() == 'TD3PlusBC':
        v_model = d3.algos.TD3PlusBC.from_json(model_json_file_name)
    elif algo.upper() == 'IQL':
        v_model = d3.algos.IQL.from_json(model_json_file_name)
    elif algo.upper() == 'MOPO':
        v_model = d3.algos.MOPO.from_json(model_json_file_name)
    elif algo.upper() == 'COMBO':
        v_model = d3.algos.COMBO.from_json(model_json_file_name)

    return v_model


def load_model_from_json_d3rlpy(algo,
                                model_json_file_name,
                                model_file_name):
    if algo.upper() == 'DDPG':
        v_model = d3.algos.DDPG.from_json(model_json_file_name)
    elif algo.upper() == 'TD3':
        v_model = d3.algos.TD3.from_json(model_json_file_name)
    elif algo.upper() == 'SAC':
        v_model = d3.algos.SAC.from_json(model_json_file_name)
    elif algo.upper() == 'BCQ':
        v_model = d3.algos.BCQ.from_json(model_json_file_name)
    elif algo.upper() == 'BEAR':
        v_model = d3.algos.BEAR.from_json(model_json_file_name)
    elif algo.upper() == 'CRR':
        v_model = d3.algos.CRR.from_json(model_json_file_name)
    elif algo.upper() == 'CQL':
        v_model = d3.algos.CQL.from_json(model_json_file_name)
    elif algo.upper() == 'AWR':
        v_model = d3.algos.AWR.from_json(model_json_file_name)
    elif algo.upper() == 'AWAC':
        v_model = d3.algos.AWAC.from_json(model_json_file_name)
    elif algo.upper() == 'PLAS':
        v_model = d3.algos.PLAS.from_json(model_json_file_name)
    elif algo.upper() == 'PLASWithPerturbation':
        v_model = d3.algos.PLASWithPerturbation.from_json(model_json_file_name)
    elif algo.upper() == 'TD3PlusBC':
        v_model = d3.algos.TD3PlusBC.from_json(model_json_file_name)
    elif algo.upper() == 'IQL':
        v_model = d3.algos.IQL.from_json(model_json_file_name)
    elif algo.upper() == 'MOPO':
        v_model = d3.algos.MOPO.from_json(model_json_file_name)
    elif algo.upper() == 'COMBO':
        v_model = d3.algos.COMBO.from_json(model_json_file_name)

    v_model.load_model(model_file_name)

    return v_model


def get_online_class_her(algo, env, learning_rate, verbose, tensorboard_log, n_sampled_goal, goal_selection_strategy,
                         online_sampling, max_episode_length):
    if algo.upper() == 'DDPG':
        v_class = DDPG(
            "MultiInputPolicy",
            env,
            learning_rate,
            tensorboard_log,
            replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            replay_buffer_kwargs=dict(
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy=goal_selection_strategy,
                online_sampling=online_sampling,
                max_episode_length=max_episode_length,
            ),
            verbose=verbose,
        )
    elif algo.upper() == 'SAC':
        v_class = SAC(
            "MultiInputPolicy",
            env,
            learning_rate,
            tensorboard_log,
            replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            replay_buffer_kwargs=dict(
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy=goal_selection_strategy,
                online_sampling=online_sampling,
                max_episode_length=max_episode_length,
            ),
            verbose=verbose,
        )
    elif algo.upper() == 'TD3':
        v_class = TD3(
            "MultiInputPolicy",
            env,
            learning_rate,
            tensorboard_log,
            replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            replay_buffer_kwargs=dict(
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy=goal_selection_strategy,
                online_sampling=online_sampling,
                max_episode_length=max_episode_length,
            ),
            verbose=verbose,
        )
    elif algo.upper() == 'DQN':
        v_class = DQN(
            "MultiInputPolicy",
            env,
            learning_rate,
            tensorboard_log,
            replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            replay_buffer_kwargs=dict(
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy=goal_selection_strategy,
                online_sampling=online_sampling,
                max_episode_length=max_episode_length,
            ),
            verbose=verbose,
        )

    return v_class


def get_action_noise_class(online_algorithm, action_noise, n_actions, v_noise_sigma):
    if online_algorithm == 'A2C' or online_algorithm == 'PPO' or online_algorithm == 'DQN':
        v_action_noise = None
    elif online_algorithm == 'DDPG' or online_algorithm == 'TD3' or online_algorithm == 'SAC':
        if action_noise == 'None':
            v_action_noise = None
        elif action_noise == 'NormalActionNoise':
            v_action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=v_noise_sigma * np.ones(n_actions))
        elif action_noise == 'OrnsteinUhlenbeckActionNoise':
            v_action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                          sigma=v_noise_sigma * np.ones(n_actions))

    return v_action_noise


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value:
    :return: current learning rate depending on remaining progress
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


def get_sorted_models(models_dir, files_lookback_hours, include_patterns):
    v_subfolders = [f.path for f in os.scandir(models_dir) if
                    f.is_dir() and _ts_to_dt(f.stat().st_ctime) > (
                            datetime.now() - timedelta(hours=files_lookback_hours))
                    # and any(str(f) in string for string in INCLUDE_PATTERNS)]
                    and [ele for ele in include_patterns if (ele in str(f))]]

    v_models = [x.split('\\')[-1] for x in v_subfolders]
    v_sorted_models = sorted(v_models, key=lambda x: x[-8:])

    return v_sorted_models


def get_patterns_list(stats_file, head):
    df = pd.read_excel(stats_file, usecols=[0], skiprowslist=[0])
    models_list = df.head(head).values.flatten().tolist()
    patterns_list = [str(x)[-8:] for x in models_list]

    return patterns_list


def get_models_and_stats_lists(stats_file, params):
    # df_stats = pd.read_excel(stats_file)
    df_stats = pd.read_csv(stats_file)
    df_stats['algo'] = df_stats.apply(parse_model_name_online_algorithm_row, args='-', axis=1)
    if params['algos'] != ['None']:
        df_stats_filtered = df_stats[df_stats['algo'].isin(params['algos'])]
    else:
        df_stats_filtered = df_stats
    df_stats_filtered['sort_column'] = df_stats_filtered[params['sort_columns']].astype(str).str.strip('%').astype(
        float)
    df_stats_sorted = df_stats_filtered.sort_values(params['sort_columns'], ascending=False)
    models_list = df_stats_sorted.head(params['head']).iloc[:, 0].values.tolist()
    if params['stat_column'] != ['None']:
        stats_list = df_stats_sorted[params['stat_column']].head(params['head']).values.tolist()
    else:
        stats_list = None

    return models_list, stats_list


def get_models_and_stats_lists_from_position(stats_file, params):
    # df_stats = pd.read_excel(stats_file)
    df_stats = pd.read_csv(stats_file)
    #   df_stats['algo'] = df_stats.apply(parse_model_name_online_algorithm_row, args='-', axis=1)
    if params['algos'] != ['None']:
        df_stats_filtered = df_stats[df_stats['algo'].isin(params['algos'])]
    else:
        df_stats_filtered = df_stats
    df_stats_filtered['sort_column'] = df_stats_filtered[params['sort_columns']].astype(str).str.strip('%').astype(
        float)
    df_stats_sorted = df_stats_filtered.sort_values(params['sort_columns'], ascending=False)
    df_stats_sorted_from_position = df_stats_sorted.iloc[params['position']:, :]
    models_list = df_stats_sorted_from_position.iloc[:, 0].head(params['number']).values.tolist()
    if params['stat_column'] != ['None']:
        stats_list = df_stats_sorted_from_position[params['stat_column']].head(params['number']).values.tolist()
    else:
        stats_list = None

    return models_list, stats_list


def get_models_and_stats_lists_for_algos(stats_file, params):
    # df_stats = pd.read_excel(stats_file)
    all_models_list, all_stats_list = [], []
    df_stats = pd.read_csv(stats_file)
    df_stats['algo'] = df_stats.apply(parse_model_name_online_algorithm_row, args='-', axis=1)
    if params['algos'] == ['None']:
        return all_models_list, all_stats_list
    else:
        for algo in params['algos']:
            print(f'Algo: {algo}')
            df_stats_filtered = df_stats[df_stats['algo'] == algo]
            df_stats_filtered['sort_column'] = df_stats_filtered[params['sort_columns']].astype(str).str.strip(
                '%').astype(float)
            df_stats_sorted = df_stats_filtered.sort_values(params['sort_columns'], ascending=False)
            models_list = df_stats_sorted.head(params['head']).iloc[:, 0].values.tolist()
            if params['stat_column'] != ['None']:
                stats_list = df_stats_sorted[params['stat_column']].head(params['head']).values.tolist()
            else:
                stats_list = None

            all_models_list.extend(models_list)
            all_stats_list.extend(stats_list)

    return all_models_list, all_stats_list


def get_top_models(stats_file, params):
    # df_stats = pd.read_excel(stats_file)
    df_stats = pd.read_csv(stats_file)
    df_stats['algo'] = df_stats.apply(parse_model_name_online_algorithm_row, args='-', axis=1)
    if params['algos'] != ['None']:
        df_stats_filtered = df_stats[df_stats['algo'].isin(params['algos'])]
    else:
        df_stats_filtered = df_stats
    df_stats_filtered['sort_column'] = df_stats_filtered[params['sort_columns']].astype(str).str.strip('%').astype(
        float)
    df_stats_sorted = df_stats_filtered.sort_values(params['sort_columns'], ascending=False)
    models_list = df_stats_sorted.head(params['head']).iloc[:, 0].values.tolist()

    return models_list
