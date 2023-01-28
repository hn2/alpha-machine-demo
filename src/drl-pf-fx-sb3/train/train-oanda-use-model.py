#   https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
#   https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
#   https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
#   https://github.com/araffin/rl-baselines-zoo

import os
import random
from os.path import join as path_join

import numpy as np
from d3rlpy.wrappers.sb3 import to_mdp_dataset
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from findrl.SaveToDiskOnBestTrainingRewardCallback import SaveToDiskOnBestTrainingRewardCallback
#   sys.path.append("../findrl")  # Adds higher directory to python modules path.
from findrl.config_utils import get_paths_params, get_pf_fx_env_params, get_train_params, \
    get_agent_params, get_model_params
from findrl.data_utils import prepare_data_train
from findrl.env_utils import make_env_pf_fx
from findrl.file_utils import get_top_models
from findrl.forex_utils import get_forex_7, get_forex_12, get_forex_14, get_forex_18, get_forex_28
from findrl.model_utils import parse_model_name_full, parse_noise_from_prefix, get_model_name, \
    get_online_class_and_policy, get_action_noise_class

DELIMITER = '-'
CONFIG_FILE = 'settings/config-oanda-train-test.ini'
# ALGOS = ['sac', 'ppo', 'a2c', 'td3', 'ddpg']
ALGOS = ['sac']
STATS_FILE = r'C:\alpha-machine\src\drl-pf-fx-sb3\stats\statistics_leverage_30-prod-Oanda-OandaBrokerage-daily-9-1-2022-0-100-.csv'


# model_to_use = 'fx_sb3_leverage_20_train_with_callback_with_random_episode_start_noise_none-7-100100-1000-30-2-oanda-daily-on_algo.sac-comp_pos.long_and_short-comp_ind.all-comp_rew.[log_returns]-true-292b203e'
# model_to_use = 'fx_sb3_leverage_20_train_with_callback_with_random_episode_start_noise_none-7-100100-1000-30-2-oanda-daily-on_algo.sac-comp_pos.long_and_short-comp_ind.all-comp_rew.[log_returns]-true-1ca309f7'
# model_to_use = 'fx_sb3_leverage_20_train_with_callback_with_random_episode_start_noise_none-7-100100-1000-30-2-oanda-daily-on_algo.sac-comp_pos.long_and_short-comp_ind.all-comp_rew.[log_returns]-true-dcd1669a'


def run_trial(model_to_use, total_timesteps, subdir, train_test_split, CONFIG_FILE):
    v_main_dir, v_models_dir, v_logs_dir, v_data_dir = get_paths_params(CONFIG_FILE)
    v_delimeter, v_model_prefix = get_model_params(CONFIG_FILE)
    v_model_verbose, v_callback_verbose, v_save_replay_buffer, v_use_tensorboard, v_use_callback, v_check_freq, \
    v_callback_lookback, v_save_freq, v_ppo_params, v_a2c_params, v_td3_params, v_sac_params, v_net_arch, v_use_linear_schedule, v_action_noise, v_noise_sigma, v_use_sde = get_agent_params(
        CONFIG_FILE)
    v_env_lookback_period, v_random_episode_start, v_cash, v_max_slippage_percent, v_lot_size, v_leverage, \
    v_compute_position, v_compute_indicators, v_compute_reward, v_meta_rl, v_env_verbose = get_pf_fx_env_params(
        CONFIG_FILE)
    v_model_prefix_used, v_number_of_instruments_used, v_total_timesteps_used, v_train_lookback_period_used, v_env_lookback_period_used, v_spread_used, v_market_used, v_resolution_used, v_online_algorithm_used, v_compute_position_used, v_compute_indicators_used, v_compute_reward_used, v_meta_rl_used = parse_model_name_full(
        model_to_use, DELIMITER)

    if v_number_of_instruments_used == 4:
        v_instruments_used, v_pip_size_used, v_pip_spread_used = get_forex_7(v_spread_used)
    elif v_number_of_instruments_used == 7:
        v_instruments_used, v_pip_size_used, v_pip_spread_used = get_forex_7(v_spread_used)
    elif v_number_of_instruments_used == 12:
        v_instruments_used, v_pip_size_used, v_pip_spread_used = get_forex_12(v_spread_used)
    elif v_number_of_instruments_used == 14:
        v_instruments_used, v_pip_size_used, v_pip_spread_used = get_forex_14(v_spread_used)
    elif v_number_of_instruments_used == 18:
        v_instruments_used, v_pip_size_used, v_pip_spread_used = get_forex_18(v_spread_used)
    elif v_number_of_instruments_used == 28:
        v_instruments_used, v_pip_size_used, v_pip_spread_used = get_forex_28(v_spread_used)

    # v_subdir, v_train_test_split, v_env_verbose, v_model_verbose, v_callback_verbose, v_save_replay_buffer, \
    # v_tensorboard, v_use_callback, v_check_freq, v_callback_lookback, v_save_freq = get_train_params(CONFIG_FILE)

    v_data = prepare_data_train(v_data_dir, subdir, v_market_used, v_resolution_used, v_instruments_used,
                                int(v_train_lookback_period_used),
                                train_test_split)

    # v_data = prepare_data_train(v_market, v_resolution, v_instruments, v_train_look_back_period)

    print(f'Data shape:{np.shape(v_data)}')

    # v_compute_position = random.choice(v_compute_position)

    print(f'v_meta_rl_used: {v_meta_rl_used}')

    v_env = make_env_pf_fx(v_data,
                           v_instruments_used,
                           v_env_lookback_period_used,
                           v_random_episode_start,
                           v_cash,
                           v_max_slippage_percent,
                           v_lot_size,
                           v_leverage,
                           v_pip_size_used,
                           v_pip_spread_used,
                           v_compute_position_used,
                           v_compute_indicators_used,
                           v_compute_reward_used,
                           v_meta_rl_used,
                           v_env_verbose)

    print(
        f'Instruments:{v_instruments_used}, lookack:{v_env_lookback_period}, random_episode_start:{v_random_episode_start}, cash:{v_cash}, max_slippage_percent:{v_max_slippage_percent}, lot_size:{v_lot_size}, leverage:{v_leverage}, pip_size:{v_pip_size_used}, pip_spread:{v_pip_spread_used}, compute_position:{v_compute_position}, compute_indicators:{v_compute_indicators}, compute_reward:{v_compute_reward}, meta_rl:{v_meta_rl}, verbose:{v_env_verbose}')

    v_model_name = get_model_name(v_delimeter, v_model_prefix, v_leverage, v_action_noise, v_use_callback,
                                  v_random_episode_start, v_instruments_used, v_total_timesteps_used,
                                  v_train_lookback_period_used,
                                  v_env_lookback_period, v_spread_used, v_market_used, v_resolution_used,
                                  v_online_algorithm_used.lower(),
                                  v_compute_position, v_compute_indicators, v_compute_reward, v_meta_rl)

    print(f'Model name:{v_model_name}')

    v_online_model_dir = path_join(*[v_models_dir, v_resolution_used, subdir, v_model_name, 'online'])
    # v_online_model_dir = path_join(v_online_models_dir, v_online_algorithm.lower())
    v_online_model_file_name = path_join(v_online_model_dir, 'model.zip')
    v_online_model_file_name_stats = path_join(v_online_model_dir, 'stats.pkl')
    v_online_model_replay_buffer = path_join(v_online_model_dir, 'replay_buffer.pkl')
    v_online_model_dataset_file_name = path_join(v_online_model_dir, 'dataset.h5')

    if not os.path.exists(v_online_model_dir):
        os.makedirs(v_online_model_dir.lower())

    v_monitor = path_join(v_logs_dir, v_model_name)
    v_dummy_vec_env = DummyVecEnv([lambda: Monitor(v_env, v_monitor)])
    v_dummy_vec_env.seed(1)

    v_vec_normalize = VecNormalize(v_dummy_vec_env)
    v_vec_normalize.seed(1)

    # v_online_algorithm = parse_model_to_use_online_algorithm(model_to_use)
    online_class, online_policy = get_online_class_and_policy(v_online_algorithm_used)

    v_action_noise = parse_noise_from_prefix(v_model_prefix_used)

    n_actions = v_dummy_vec_env.action_space.shape[-1]

    v_action_noise_class = get_action_noise_class(v_online_algorithm_used.upper(), v_action_noise.capitalize(),
                                                  n_actions, v_noise_sigma)

    print(f'Using model: {model_to_use}')
    print(f'Setting model paramaters ...')
    print(online_class)
    v_online_model_used = online_class(online_policy, v_vec_normalize, verbose=1)
    v_online_model_used.load(path_join(*[v_models_dir, v_resolution_used, subdir, model_to_use, 'online', 'model.zip']))

    print(f'Online model paramaters ...')
    print(v_online_model_used.get_parameters())
    print(v_online_model_used.policy_kwargs)
    print(v_online_model_used._total_timesteps)

    print("Start training model...")

    try:

        if v_use_callback:
            callback = SaveToDiskOnBestTrainingRewardCallback(check_freq=v_check_freq, save_freq=v_save_freq,
                                                              lookback=v_callback_lookback,
                                                              online_algorithm=v_online_algorithm_used,
                                                              model_file_name=v_online_model_file_name,
                                                              model_replay_buffer=v_online_model_replay_buffer,
                                                              model_stats=v_online_model_file_name_stats,
                                                              save_replay_buffer=v_save_replay_buffer,
                                                              verbose=v_callback_verbose)
            v_online_model_used.learn(total_timesteps=total_timesteps, log_interval=1000, reset_num_timesteps=False,
                                      tb_log_name=v_model_name, callback=callback)
        else:
            v_online_model_used.learn(total_timesteps=total_timesteps, log_interval=1000, reset_num_timesteps=False,
                                      tb_log_name=v_model_name)

        if not v_use_callback:
            v_online_model_used.save(v_online_model_file_name.lower())
            v_vec_normalize.save(v_online_model_file_name_stats.lower())

    except Exception as e:
        print(e)

    if v_save_replay_buffer:
        try:
            dataset = to_mdp_dataset(v_online_model_used.replay_buffer)
            dataset.dump(v_online_model_dataset_file_name)
            os.remove(v_online_model_replay_buffer.lower())
            # os.remove(v_online_model_file_name)
            # os.remove(v_online_model_file_name_stats)
        except Exception as e:
            print(e)

    print("End training online model...")

    v_env.close()


def main():
    # CONFIG_FILE = 'settings/config-oanda.ini'

    v_algorithms, v_number_of_trials, v_subdir, v_train_test_split, v_train_look_back_period, v_total_timesteps, v_market, v_resolution, v_number_of_instruments, v_spread, v_num_cpu = get_train_params(
        CONFIG_FILE)

    v_models_to_use = []

    print(ALGOS)

    for algo in ALGOS:
        params = {
            "algos": [
                algo
            ],
            "sort_columns": "Expectancy",
            "head": 3
        }

        v_models = get_top_models(STATS_FILE, params)

        v_models_to_use.extend(v_models)

    print(f'Models to use: {v_models_to_use}')

    for i in range(v_number_of_trials):
        v_model_to_use = random.choice(v_models_to_use)
        run_trial(v_model_to_use, v_total_timesteps, v_subdir, v_train_test_split, CONFIG_FILE)


if __name__ == "__main__":
    main()
