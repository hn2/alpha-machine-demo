#   https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
#   https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
#   https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
#   https://github.com/araffin/rl-baselines-zoo
#   https://stackoverflow.com/questions/20716812/saving-and-loading-multiple-objects-in-pickle-file

#   sudo apt -y install nvidia-cuda-toolkit
#   lspci | grep -i nvidia
#   python -c 'import torch; print(torch.cuda.is_available())'

#   cd c:\alpha-machine\src\drl-pf-fx-sb3\train & c:
#   cd /home/ubuntu/alpha-machine/src/drl-pf-fx-sb3/train
#   Example of usage:
#   python train-oanda.py -R daily -A td3 -D false -N 100
#   python train-oanda.py -R daily -A ddpg td3 -D false -N 100
#   python train-oanda.py -R hour -A td3 -D false -N 100
#   python train-oanda.py -R hour -A ddpg td3 -D false -N 100
#   python train-oanda.py -R daily -A a2c ppo ddpg td3 sac -D false -N 100
#   python train-oanda.py -R hour -A a2c ppo ddpg td3 -D false -N 100
#   python train-oanda.py -R daily -A td3 -D false -N 100
#   python train-oanda.py -R daily -A random -D false -N 100
#   python train-oanda.py -R daily -A random -D true -N 100
#   python train-oanda.py -R daily -A td3 -D false -N 100
#   python train-oanda.py -R hour -A ddpg -D false -N 100
#   python train-oanda.py -R hour -A random -D false -N 100
#   python train-oanda.py -R hour -A random -D true -N 100


import argparse
import copy
import os
import platform
import random
from datetime import datetime
from distutils.util import strtobool
from os.path import join as path_join

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from findrl.SaveToDiskOnBestTrainingRewardCallback import SaveToDiskOnBestTrainingRewardCallback
from findrl.config_utils import get_paths_windows_params, get_paths_linux_params, get_pf_fx_env_params_train, \
    get_train_params, get_agent_params_sb3, get_model_params
from findrl.data_utils import prepare_data_train_with_offset  # , calculate_features_new
from findrl.datetime_utils import WEEKDAY, get_week_number
from findrl.env_utils import make_env_pf_fx
from findrl.file_utils import dump_dict_to_pickle
from findrl.forex_utils import check_convert
from findrl.forex_utils import get_instruments_in_portfolio, get_instruments_in_portfolio_fixed
from findrl.model_utils import get_model_name, get_online_class_and_policy_sb3, get_action_noise_class, linear_schedule


def prepare_env(data_dir, market, resolution, train_look_back_period, offset, account_currency,
                instruments_in_portfolio, pip_size_in_portfolio, pip_spread_in_portfolio,
                env_lookback_period, random_episode_start, cash, max_slippage_percent, lot_size,
                leverage, compute_position, compute_indicators, compute_reward, meta_rl, env_verbose):
    v_data = prepare_data_train_with_offset(data_dir, market, resolution, instruments_in_portfolio,
                                            train_look_back_period, offset)

    print(f'Data shape:{np.shape(v_data)}')

    #   v_compute_position = random.choice(v_compute_position)

    v_env = make_env_pf_fx(v_data,
                           account_currency,
                           instruments_in_portfolio,
                           env_lookback_period,
                           random_episode_start,
                           cash,
                           max_slippage_percent,
                           lot_size,
                           leverage,
                           pip_size_in_portfolio,
                           pip_spread_in_portfolio,
                           compute_position,
                           compute_indicators,
                           compute_reward,
                           meta_rl,
                           env_verbose)

    print(
        f'Instruments:{instruments_in_portfolio}, lookack:{env_lookback_period}, random_episode_start:{random_episode_start}, cash:{cash}, max_slippage_percent:{max_slippage_percent}, lot_size:{lot_size}, leverage:{leverage}, compute_position:{compute_position}, compute_indicators:{compute_indicators}, compute_reward:{compute_reward}, meta_rl:{meta_rl}, verbose:{env_verbose}')

    return v_env


def run_trial(env, is_default_params, models_dir, resolution, subdir, model_name, logs_dir, online_algorithm,
              action_noise, noise_sigma, params, learning_rate, use_linear_schedule, use_tensorboard, model_verbose,
              use_callback, check_freq, save_freq, callback_lookback, save_replay_buffer, callback_verbose,
              total_timesteps, log_interval):
    v_online_model_dir = path_join(*[models_dir, resolution, subdir, model_name, 'online'])

    if not os.path.exists(v_online_model_dir):
        os.makedirs(v_online_model_dir)

    #   Save env attributes without data
    v_online_model_env_file_name = path_join(v_online_model_dir, 'env.pkl')
    v_env_attributes = dict(env.__dict__)
    v_PfFxEnv_attributes = v_env_attributes['env']
    v_PfFxEnv_attributes_to_save = copy.deepcopy(v_PfFxEnv_attributes)
    v_PfFxEnv_attributes_to_save.data = None
    v_PfFxEnv_attributes_to_save.price_data = None
    v_PfFxEnv_attributes_to_save.features_data = None
    dump_dict_to_pickle(v_online_model_env_file_name, v_PfFxEnv_attributes_to_save)

    v_online_model_file_name = path_join(v_online_model_dir, 'model.zip')
    v_online_model_file_name_stats = path_join(v_online_model_dir, 'stats.pkl')
    v_online_model_replay_buffer = path_join(v_online_model_dir, 'replay_buffer')
    v_online_model_dataset_file_name = path_join(v_online_model_dir, 'dataset.h5')

    print(f'v_online_model_file_name: {v_online_model_file_name}')
    print(f'v_online_model_file_name_stats: {v_online_model_file_name_stats}')

    v_monitor = path_join(logs_dir, model_name)
    v_dummy_vec_env = DummyVecEnv([lambda: Monitor(env, v_monitor)])
    #   v_dummy_vec_env = DummyVecEnv([lambda: env])
    v_dummy_vec_env.seed(1)

    v_online_class, v_online_policy = get_online_class_and_policy_sb3(online_algorithm)

    print(f'Online class: {v_online_class}, Online policy: {v_online_policy}')

    n_actions = v_dummy_vec_env.action_space.shape[-1]
    v_action_noise_class = get_action_noise_class(online_algorithm, action_noise, n_actions, noise_sigma)

    # load recent checkpoint
    if os.path.isfile(v_online_model_file_name) and os.path.isfile(v_online_model_file_name_stats):
        v_vec_normalize = VecNormalize.load(v_online_model_file_name_stats, v_dummy_vec_env)
        v_vec_normalize.reset()
        v_online_model = v_online_class.load(v_online_model_file_name, v_vec_normalize)
        print('Model Loaded ...')
    else:
        #   v_vec_normalize = VecNormalize(v_dummy_vec_env, norm_obs, norm_reward, clip_obs, clip_reward, gamma)
        v_vec_normalize = VecNormalize(v_dummy_vec_env)

    v_vec_normalize.seed(1)

    if is_default_params:
        v_online_model = v_online_class(env=v_vec_normalize, policy=v_online_policy,
                                        tensorboard_log=logs_dir if use_tensorboard else None,
                                        verbose=model_verbose)
    else:
        if online_algorithm == 'SAC' or online_algorithm == 'DDPG' or online_algorithm == 'TD3':
            v_online_model = v_online_class(env=v_vec_normalize, policy=v_online_policy, **params,
                                            action_noise=v_action_noise_class,
                                            learning_rate=linear_schedule(
                                                learning_rate) if use_linear_schedule else learning_rate,
                                            tensorboard_log=logs_dir if use_tensorboard else None,
                                            verbose=model_verbose)
        else:
            v_online_model = v_online_class(env=v_vec_normalize, policy=v_online_policy, **params,
                                            learning_rate=linear_schedule(
                                                learning_rate) if use_linear_schedule else learning_rate,
                                            tensorboard_log=logs_dir if use_tensorboard else None,
                                            verbose=model_verbose)

    # replay buffer
    if os.path.isfile(v_online_model_replay_buffer):
        v_online_model.load_replay_buffer(v_online_model_replay_buffer)

    print(
        f'Start training model with account_currency {v_PfFxEnv_attributes.account_currency}, instrument {v_PfFxEnv_attributes.instruments}, pip_size {v_PfFxEnv_attributes.pip_size}, pip_spread {v_PfFxEnv_attributes.pip_spread}...')

    try:

        if use_callback:
            callback = SaveToDiskOnBestTrainingRewardCallback(check_freq=check_freq, save_freq=save_freq,
                                                              lookback=callback_lookback,
                                                              online_algorithm=online_algorithm,
                                                              model_file_name=v_online_model_file_name,
                                                              model_replay_buffer=v_online_model_replay_buffer,
                                                              model_stats=v_online_model_file_name_stats,
                                                              save_replay_buffer=save_replay_buffer,
                                                              verbose=callback_verbose)
            v_online_model.learn(total_timesteps=total_timesteps, log_interval=log_interval, reset_num_timesteps=False,
                                 tb_log_name=model_name, callback=callback)
        else:
            v_online_model.learn(total_timesteps=total_timesteps, log_interval=log_interval, reset_num_timesteps=False,
                                 tb_log_name=model_name)

        if not use_callback:
            v_online_model.save(v_online_model_file_name)
            v_vec_normalize.save(v_online_model_file_name_stats)

    except Exception as e:
        print(e)

    if save_replay_buffer:
        try:
            v_online_model.save_replay_buffer(v_online_model_replay_buffer)
        except Exception as e:
            print(e)

    print("End training online model...")

    env.close()


def main(config_file, config_paths, is_default_params):
    if platform.system() == 'Windows':
        v_files_dir, v_main_dir, v_main_code_dir, v_main_code_dir_train, v_main_code_dir_test, v_main_code_dir_tune, v_models_dir, v_logs_dir, v_data_dir, v_stats_dir = get_paths_windows_params(
            config_paths)
    elif platform.system() == 'Linux':
        v_files_dir, v_main_dir, v_main_code_dir, v_main_code_dir_train, v_main_code_dir_test, v_main_code_dir_tune, v_models_dir, v_logs_dir, v_data_dir, v_stats_dir = get_paths_linux_params(
            config_paths)
    v_delimeter, v_model_prefix = get_model_params(config_file)
    v_algorithms_list, v_model_verbose, v_callback_verbose, v_save_replay_buffer, v_use_tensorboard, v_use_callback, v_check_freq, \
    v_callback_lookback, v_save_freq, v_params, v_learning_rates, v_batch_sizes, v_net_arch, v_use_linear_schedule, v_action_noises, v_noise_sigma, v_use_sdes = get_agent_params_sb3(
        config_file)
    v_env_lookback_periods, v_random_episode_start, v_cash, v_max_slippage_percent, v_lot_size, v_leverage, \
    v_compute_position, v_compute_indicatorss, v_compute_rewards, v_meta_rls, v_env_verbose, v_num_of_envs = get_pf_fx_env_params_train(
        config_file)
    v_number_of_trials, v_subdir, v_train_look_back_periods, v_offset, v_total_timesteps, v_log_interval, v_market, \
    v_resolution, v_instruments_in_portfolio_list, v_num_of_instruments, v_num_of_instruments_in_portfolio, v_is_equal, v_spread = get_train_params(
        config_file)

    today = datetime.today()
    WEEK = get_week_number(WEEKDAY.FRI, today)
    YEAR = today.year

    #   v_total_timesteps = v_total_timesteps + WEEK

    #   for i in range(v_number_of_trials):

    v_online_algorithm = random.choice(v_algorithms_list)
    v_action_noise = random.choice(v_action_noises)
    v_use_sde = random.choices(v_use_sdes)
    v_batch_size = random.choice(v_batch_sizes)
    v_env_lookback_period = random.choice(v_env_lookback_periods)
    v_compute_indicators = random.choice(v_compute_indicatorss)
    v_compute_reward = random.choices(v_compute_rewards)
    v_train_look_back_period = random.choice(v_train_look_back_periods)
    v_learning_rate = random.choice(v_learning_rates)
    v_meta_rl = bool(strtobool(random.choice(v_meta_rls)))

    if v_online_algorithm == 'PPO' or v_online_algorithm == 'DDPG' or v_online_algorithm == 'DQN' or v_online_algorithm == 'SAC' or v_online_algorithm == 'TD3':
        v_params['batch_size'] = v_batch_size
    elif v_online_algorithm == 'A2C':
        v_params['n_steps'] = v_batch_size

    if v_online_algorithm == 'A2C' or v_online_algorithm == 'PPO' or v_online_algorithm == 'SAC':
        v_params['use_sde'] = v_use_sde

    v_params['policy_kwargs'] = dict(net_arch=v_net_arch)

    if is_default_params:
        print('Using default params')
    else:
        print(f'Using {v_params}')

    if v_instruments_in_portfolio_list:
        v_instruments_in_portfolio = random.choice(v_instruments_in_portfolio_list)
        v_instruments_in_portfolio, v_pip_size_in_portfolio, v_pip_spread_in_portfolio = get_instruments_in_portfolio_fixed(
            v_instruments_in_portfolio, v_spread)
    else:
        v_instruments_in_portfolio, v_pip_size_in_portfolio, v_pip_spread_in_portfolio = get_instruments_in_portfolio(
            v_num_of_instruments, v_num_of_instruments_in_portfolio, v_is_equal, v_spread)

    v_account_currencies = ('usd', 'eur', 'jpy', 'gbp', 'aud', 'cad', 'chf', 'nzd')
    #   v_account_currencies = ['usd']
    v_account_currency, v_check_convert = check_convert(v_account_currencies, v_instruments_in_portfolio,
                                                        v_lot_size, v_leverage,
                                                        np.ones(len(v_instruments_in_portfolio)))

    # if not v_check_convert:
    #     continue

    print(f'Account currency: {v_account_currency}')
    print(f'Offset: {v_offset}')

    v_env = prepare_env(v_data_dir, v_market, v_resolution, v_train_look_back_period, v_offset, v_account_currency,
                        v_instruments_in_portfolio, v_pip_size_in_portfolio, v_pip_spread_in_portfolio,
                        v_env_lookback_period, v_random_episode_start, v_cash, v_max_slippage_percent, v_lot_size,
                        v_leverage, v_compute_position, v_compute_indicators, v_compute_reward, v_meta_rl,
                        v_env_verbose)

    v_learning_rate_formatted = f'{v_learning_rate:.5f}'

    if v_online_algorithm == 'PPO' or v_online_algorithm == 'DDPG' or v_online_algorithm == 'DQN' or v_online_algorithm == 'SAC' or v_online_algorithm == 'TD3':
        if is_default_params:
            v_model_prefix = f'{v_model_prefix}_wk.{str(WEEK)}_default_params_'
        else:
            v_model_prefix = f'{v_model_prefix}_wk.{str(WEEK)}_lr.{str(v_learning_rate_formatted)}_b.{str(v_batch_size)}_'
    elif v_online_algorithm == 'A2C':
        if is_default_params:
            v_model_prefix = f'{v_model_prefix}_wk.{str(WEEK)}_default_params_'
        else:
            v_model_prefix = f'{v_model_prefix}_wk.{str(WEEK)}_lr.{str(v_learning_rate_formatted)}_s.{str(v_batch_size)}_'

    if v_online_algorithm == 'A2C' or v_online_algorithm == 'PPO' or v_online_algorithm == 'SAC':
        v_model_prefix = v_model_prefix + f'sde.{v_use_sde[0]}_'

    print('-----------')
    print(v_action_noises)
    print(v_action_noise)

    if v_online_algorithm == 'DDPG' or v_online_algorithm == 'TD3' or v_online_algorithm == 'SAC':
        if v_action_noise == "OrnsteinUhlenbeckActionNoise":
            v_model_prefix = v_model_prefix + 'ns.ou_'
        elif v_action_noise == "NormalActionNoise":
            v_model_prefix = v_model_prefix + 'ns.normal_'
        elif v_action_noise == "None":
            v_model_prefix = v_model_prefix + 'ns.none_'

    v_model_prefix = v_model_prefix + f'cb.{v_use_callback}_res.{v_random_episode_start}_lev.{v_leverage}' + v_delimeter

    v_model_name = get_model_name(v_delimeter, v_model_prefix, len(v_instruments_in_portfolio), v_total_timesteps,
                                  v_train_look_back_period, v_offset, v_env_lookback_period, v_spread, v_market,
                                  v_resolution, v_online_algorithm.lower(), v_compute_position, v_compute_indicators,
                                  v_compute_reward[0], v_meta_rl)

    print(f'Models dir:{v_models_dir}')
    print(f'Model name:{v_model_name}')

    run_trial(v_env, is_default_params, v_models_dir, v_resolution, v_subdir, v_model_name, v_logs_dir,
              v_online_algorithm, v_action_noise, v_noise_sigma, v_params, v_learning_rate, v_use_linear_schedule,
              v_use_tensorboard, v_model_verbose, v_use_callback, v_check_freq, v_save_freq,
              v_callback_lookback, v_save_replay_buffer, v_callback_verbose, v_total_timesteps, v_log_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train oanda')
    # parser.add_argument('-C', '--config_file', help='Name of config file', required=True)
    parser.add_argument('-R', '--resolution', help='hour, daily, random', default='daily', type=str, required=True)
    parser.add_argument('-A', '--algo', help='a2c, ppo, sac, ddpg, td3, random', nargs="+", default=["ddpg", "td3,"],
                        type=str, required=True)
    # parser.add_argument('-D', '--default_params', help='true, false, random', default='false',
    #                     type=lambda x: bool(strtobool(x)), type=ascii, required=True)
    parser.add_argument('-D', '--default_params', help='true, false, random', default='false',
                        type=str, required=True)
    parser.add_argument('-N', '--number_of_trials', help='100', default=100, type=int, required=True)
    args = vars(parser.parse_args())
    # arg_config_file = args['config_file']
    arg_number_of_trials = args['number_of_trials']
    #   number_of_trials = int(arg_number_of_trials)
    for i in range(arg_number_of_trials):
        arg_algo = args['algo']
        print(arg_algo)
        algo = random.choices(arg_algo)[0]
        print(algo)
        arg_resolution = args['resolution']
        if arg_resolution.lower() == 'random':
            resolution = random.choices(['hour', 'daily'])
        else:
            resolution = arg_resolution.lower()
        print(resolution)
        arg_default_params = args['default_params']
        if arg_default_params == 'random':
            default_params = bool(strtobool(random.choices(['true', 'false'])))
        else:
            default_params = bool(strtobool(arg_default_params))
        print(default_params)
        config_file = f'../settings/train/config-train-oanda-{resolution}-{algo}.ini'
        config_paths = f'../settings/general/config-paths.ini'
        print(
            f'Training with config file {config_file}, config paths {config_paths} and default params {default_params}')
        main(config_file, config_paths, default_params)
