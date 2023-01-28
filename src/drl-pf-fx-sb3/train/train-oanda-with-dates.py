#   https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
#   https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
#   https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
#   https://github.com/araffin/rl-baselines-zoo
#   https://stackoverflow.com/questions/20716812/saving-and-loading-multiple-objects-in-pickle-file
#   v_num_of_lookback_bars_offset = 1000
#   v_num_of_lookback_bars = 4500
#   Daily
#   end_date = datetime.now() - timedelta(days=v_num_of_lookback_bars_offset)
#   start_date = end - timedelta(days=v_num_of_lookback_bars)
#   hour
#   end_date = datetime.now() - timedelta(days=round(v_num_of_lookback_bars_offset / 24))
#   start_date = end - timedelta(days=round(v_num_of_lookback_bars / 24))
#   cd E:\alpha-machine\src\drl-pf-fx-sb3\train & e:
#   Example of usage:
#   v_num_of_lookback_bars_offset = 1000
#   v_num_of_lookback_bars = 4500
#   python train-oanda-with-dates.py -R daily -A random -D true -S 20070919 -E 20200114 -N 100
#   python train-oanda-with-dates.py -R daily -A random -D false -S 20070919 -E 20200114 -N 100
#   python train-oanda-with-dates.py -R daily -A ddpg -D false -S 20070919 -E 20200114 -N 100
#   python train-oanda-with-dates.py -R hour -A random -D true -S 2022022 -E 20220829 -N 100
#   python train-oanda-with-dates.py -R hour -A random -D false -S 2022022 -E 20220829 -N 100
#   python train-oanda-with-dates.py -R hour -A ddpg -D false -S 2022022 -E 20220829 -N 100
#   v_num_of_lookback_bars_offset = 500
#   v_num_of_lookback_bars = 5000
#   python train-oanda-with-dates.py -R daily -A random -D true -S 20070919 -E 20210528 -N 100
#   python train-oanda-with-dates.py -R daily -A random -D false -S 20070919 -E 20210528 -N 100
#   python train-oanda-with-dates.py -R daily -A ddpg -D false -S 20070919 -E 20210528 -N 100
#   python train-oanda-with-dates.py -R hour -A random -D true -S 2022023 -E 20220919 -N 100
#   python train-oanda-with-dates.py -R hour -A random -D false -S 2022023 -E 20220919 -N 100
#   python train-oanda-with-dates.py -R hour -A ddpg -D false -S 2022023 -E 20220919 -N 100


import argparse
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
#   sys.path.append("../findrl")  # Adds higher directory to python modules path.
from findrl.config_utils import get_paths_windows_params, get_paths_linux_params, get_pf_fx_env_params_train, \
    get_train_params, get_agent_params_sb3, get_model_params
from findrl.data_utils import prepare_data_train_with_dates  # , calculate_features_new
from findrl.datetime_utils import WEEKDAY, get_week_number
from findrl.env_utils import make_env_pf_fx
from findrl.file_utils import dump_dict_to_pickle
from findrl.forex_utils import check_convert
from findrl.forex_utils import get_instruments_in_portfolio, get_instruments_in_portfolio_fixed
from findrl.model_utils import get_model_name, get_online_class_and_policy_sb3, get_action_noise_class, linear_schedule


def prepare_env(data_dir, market, resolution, start_date, end_date, account_currency,
                instruments_in_portfolio, pip_size_in_portfolio, pip_spread_in_portfolio,
                env_lookback_period, random_episode_start, cash, max_slippage_percent, lot_size,
                leverage, compute_position, compute_indicators, compute_reward, meta_rl, env_verbose):
    v_start_date, v_end_date, v_data = prepare_data_train_with_dates(data_dir, market, resolution,
                                                                     instruments_in_portfolio, start_date, end_date)

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
    v_PfFxEnv_attributes.data = None
    dump_dict_to_pickle(v_online_model_env_file_name, v_PfFxEnv_attributes)

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


def main(config_file, is_default_params, start_date, end_date):
    if platform.system() == 'Windows':
        v_main_dir, v_main_code_dir, v_main_code_dir_train, v_main_code_dir_test, v_main_code_dir_tune, v_models_dir, v_logs_dir, v_data_dir, v_test_dir, v_stats_dir = get_paths_windows_params(
            config_file)
    if platform.system() == 'Linux':
        v_main_dir, v_main_code_dir, v_main_code_dir_train, v_main_code_dir_test, v_main_code_dir_tune, v_models_dir, v_logs_dir, v_data_dir, v_test_dir, v_stats_dir = get_paths_linux_params(
            config_file)
    v_delimeter, v_model_prefix = get_model_params(config_file)
    v_algorithms_list, v_model_verbose, v_callback_verbose, v_save_replay_buffer, v_use_tensorboard, v_use_callback, v_check_freq, \
    v_callback_lookback, v_save_freq, v_params, v_learning_rates, v_batch_sizes, v_net_arch, v_use_linear_schedule, v_action_noises, v_noise_sigma, v_use_sde = get_agent_params_sb3(
        config_file)
    v_env_lookback_periods, v_random_episode_start, v_cash, v_max_slippage_percent, v_lot_size, v_leverage, \
    v_compute_position, v_compute_indicatorss, v_compute_reward, v_meta_rls, v_env_verbose, v_num_of_envs = get_pf_fx_env_params_train(
        config_file)
    v_number_of_trials, v_subdir, v_train_look_back_periods, v_total_timesteps, v_log_interval, v_market, \
    v_resolution, v_instruments_in_portfolio_list, v_num_of_instruments, v_num_of_instruments_in_portfolio, v_is_equal, v_spread = get_train_params(
        config_file)

    today = datetime.today()
    WEEK = get_week_number(WEEKDAY.FRI, today)
    YEAR = today.year

    #   v_total_timesteps = v_total_timesteps + WEEK

    #   for i in range(v_number_of_trials):

    v_online_algorithm = random.choice(v_algorithms_list)
    v_action_noise = random.choice(v_action_noises)
    v_batch_size = random.choice(v_batch_sizes)
    v_env_lookback_period = random.choice(v_env_lookback_periods)
    v_compute_indicators = random.choice(v_compute_indicatorss)
    v_train_look_back_period = random.choice(v_train_look_back_periods)
    v_learning_rate = random.choice(v_learning_rates)
    v_meta_rl = bool(strtobool(random.choice(v_meta_rls)))

    if v_online_algorithm == 'PPO' or v_online_algorithm == 'DDPG' or v_online_algorithm == 'DQN' or v_online_algorithm == 'SAC' or v_online_algorithm == 'TD3':
        v_params['batch_size'] = v_batch_size
    elif v_online_algorithm == 'A2C':
        v_params['n_steps'] = v_batch_size

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

    v_env = prepare_env(v_data_dir, v_market, v_resolution, start_date, end_date, v_account_currency,
                        v_instruments_in_portfolio, v_pip_size_in_portfolio, v_pip_spread_in_portfolio,
                        v_env_lookback_period, v_random_episode_start, v_cash, v_max_slippage_percent, v_lot_size,
                        v_leverage, v_compute_position, v_compute_indicators, v_compute_reward, v_meta_rl,
                        v_env_verbose)

    v_learning_rate_formatted = f'{v_learning_rate:.5f}'

    v_model_prefix_full = f'{v_model_prefix}_w_{str(WEEK)}_'

    if v_online_algorithm == 'PPO' or v_online_algorithm == 'DDPG' or v_online_algorithm == 'DQN' or v_online_algorithm == 'SAC' or v_online_algorithm == 'TD3':
        batch_size = v_params["batch_size"]
        if is_default_params:
            v_model_prefix_full = f'{v_model_prefix_full}dp_'
        else:
            v_model_prefix_full = f'{v_model_prefix_full}lr_{str(v_learning_rate_formatted)}_b_{str(batch_size)}_'
    elif v_online_algorithm == 'A2C':
        n_steps = v_params['n_steps']
        if is_default_params:
            v_model_prefix_full = f'{v_model_prefix_full}dp_'
        else:
            v_model_prefix_full = f'{v_model_prefix_full}lr_{str(v_learning_rate_formatted)}_s_{str(n_steps)}_'

    start_date_formatted = start_date.strftime('%Y%m%d')
    end_date_formatted = end_date.strftime('%Y%m%d')

    v_model_prefix_full = f'{v_model_prefix_full}{start_date_formatted}__{end_date_formatted}_'

    if v_resolution == 'daily':
        v_train_look_back_period = (end_date - start_date).days
    elif v_resolution == 'hour':
        v_train_look_back_period = (end_date - start_date).days * 24

    v_model_name = get_model_name(v_delimeter, v_model_prefix_full, v_leverage, v_action_noise, v_use_callback,
                                  v_random_episode_start, len(v_instruments_in_portfolio), v_total_timesteps,
                                  v_train_look_back_period, v_env_lookback_period, v_spread, v_market, v_resolution,
                                  v_online_algorithm.lower(), v_compute_position, v_compute_indicators,
                                  v_compute_reward, v_meta_rl)

    print(f'Model name:{v_model_name}')

    run_trial(v_env, is_default_params, v_models_dir, v_resolution, v_subdir, v_model_name, v_logs_dir,
              v_online_algorithm,
              v_action_noise, v_noise_sigma, v_params, v_learning_rate, v_use_linear_schedule,
              v_use_tensorboard, v_model_verbose, v_use_callback, v_check_freq, v_save_freq,
              v_callback_lookback, v_save_replay_buffer, v_callback_verbose, v_total_timesteps, v_log_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train oanda')
    # parser.add_argument('-C', '--config_file', help='Name of config file', required=True)
    parser.add_argument('-R', '--resolution', help='hour, daily, random', default='daily', type=str, required=True)
    parser.add_argument('-A', '--algo', help='a2c, ppo, sac, ddpg, td3, random', default='random', type=str,
                        required=True)
    # parser.add_argument('-D', '--default_params', help='true, false, random', default='false',
    #                     type=lambda x: bool(strtobool(x)), type=ascii, required=True)
    parser.add_argument('-D', '--default_params', help='true, false, random', default='false',
                        type=str, required=True)
    parser.add_argument('-N', '--number_of_trials', help='100', default=100, type=int, required=True)
    parser.add_argument('-S', '--start_date', type=lambda d: datetime.strptime(d, '%Y%m%d').date(),
                        help='Date in the format yyyymmdd')
    parser.add_argument('-E', '--end_date', type=lambda d: datetime.strptime(d, '%Y%m%d').date(),
                        help='Date in the format yyyymmdd')
    args = vars(parser.parse_args())
    # arg_config_file = args['config_file']
    arg_number_of_trials = args['number_of_trials']
    #   number_of_trials = int(arg_number_of_trials)
    start_date = args["start_date"]
    end_date = args["end_date"]
    for i in range(arg_number_of_trials):
        arg_algo = args['algo']
        print(arg_algo)
        if arg_algo.lower() == 'random':
            algo = random.choices(['a2c', 'ppo', 'sac', 'ddpg', 'td3'])[0]
        else:
            algo = arg_algo.lower()
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
        config_file = f'../settings/config-train-oanda-{resolution}-{algo}.ini'
        print(f'Training with config file {config_file} and default params {default_params}')
        main(config_file, default_params, start_date, end_date)
