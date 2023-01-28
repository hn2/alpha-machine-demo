#   https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
#   https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
#   https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
#   https://github.com/araffin/rl-baselines-zoo

import os
import random
from os.path import join as path_join

import numpy as np
import torch as th
from sac_plus.sac.sac_continuos_action import SAC

#   sys.path.append("../findrl")  # Adds higher directory to python modules path.
from findrl.config_utils import get_paths_params, get_pf_fx_env_params, get_train_params, \
    get_agent_params, get_model_params
from findrl.data_utils import prepare_data_train
from findrl.env_utils import make_env_pf_fx
from findrl.forex_utils import get_forex_7, get_forex_12, get_forex_14, get_forex_18, get_forex_28
from findrl.model_utils import get_model_name

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)

CONFIG_FILE = 'settings/config-oanda-train-test.ini'


def run_trial(online_algorithm, train_look_back_period, total_timesteps, market, resolution, num_of_instruments, spread,
              num_cpu, subdir, train_test_split, config_file):
    v_main_dir, v_models_dir, v_logs_dir, v_data_dir = get_paths_params(config_file)
    v_delimeter, v_model_prefix = get_model_params(config_file)
    v_model_verbose, v_callback_verbose, v_save_replay_buffer, v_use_tensorboard, v_use_callback, v_check_freq, \
    v_callback_lookback, v_save_freq, v_ppo_params, v_a2c_params, v_td3_params, v_sac_params, v_net_arch, v_use_linear_schedule, v_action_noise, v_noise_sigma, v_use_sde = get_agent_params(
        config_file)
    v_env_lookback_period, v_random_episode_start, v_cash, v_max_slippage_percent, v_lot_size, v_leverage, \
    v_compute_position, v_compute_indicators, v_compute_reward, v_meta_rl, v_env_verbose = get_pf_fx_env_params(
        config_file)

    v_online_algorithm = random.choice(online_algorithm)
    v_num_of_instruments = int(random.choice(num_of_instruments))
    v_total_timesteps = total_timesteps

    if v_num_of_instruments == 4:
        v_instruments, v_pip_size, v_pip_spread = get_forex_7(spread)
    elif v_num_of_instruments == 7:
        v_instruments, v_pip_size, v_pip_spread = get_forex_7(spread)
    elif v_num_of_instruments == 12:
        v_instruments, v_pip_size, v_pip_spread = get_forex_12(spread)
    elif v_num_of_instruments == 14:
        v_instruments, v_pip_size, v_pip_spread = get_forex_14(spread)
    elif v_num_of_instruments == 18:
        v_instruments, v_pip_size, v_pip_spread = get_forex_18(spread)
    elif v_num_of_instruments == 28:
        v_instruments, v_pip_size, v_pip_spread = get_forex_28(spread)

    # v_subdir, v_train_test_split, v_env_verbose, v_model_verbose, v_callback_verbose, v_save_replay_buffer, \
    # v_tensorboard, v_use_callback, v_check_freq, v_callback_lookback, v_save_freq = get_train_params(config_file)

    v_data = prepare_data_train(v_data_dir, subdir, market, resolution, v_instruments, train_look_back_period,
                                train_test_split)

    # v_data = prepare_data_train(v_market, v_resolution, v_instruments, v_train_look_back_period)

    print(f'Data shape:{np.shape(v_data)}')

    v_compute_position = random.choice(v_compute_position)

    v_env = make_env_pf_fx(v_data,
                           v_instruments,
                           v_env_lookback_period,
                           v_random_episode_start,
                           v_cash,
                           v_max_slippage_percent,
                           v_lot_size,
                           v_leverage,
                           v_pip_size,
                           v_pip_spread,
                           v_compute_position,
                           v_compute_indicators,
                           v_compute_reward,
                           v_meta_rl,
                           v_env_verbose)
    v_env.seed(SEED)

    print(
        f'Instruments:{v_instruments}, lookack:{v_env_lookback_period}, random_episode_start:{v_random_episode_start}, cash:{v_cash}, max_slippage_percent:{v_max_slippage_percent}, lot_size:{v_lot_size}, leverage:{v_leverage}, pip_size:{v_pip_size}, pip_spread:{v_pip_spread}, compute_position:{v_compute_position}, compute_indicators:{v_compute_indicators}, compute_reward:{v_compute_reward}, meta_rl:{v_meta_rl}, verbose:{v_env_verbose}')

    v_model_name = get_model_name(v_delimeter, v_model_prefix, v_leverage, v_action_noise, v_use_callback,
                                  v_random_episode_start, v_instruments, v_total_timesteps, train_look_back_period,
                                  v_env_lookback_period, spread, market, resolution, v_online_algorithm.lower(),
                                  v_compute_position, v_compute_indicators, v_compute_reward, v_meta_rl)

    print(f'Model name:{v_model_name}')

    v_online_model_dir = path_join(*[v_models_dir, resolution, subdir, v_model_name, 'online'])
    # v_online_model_dir = path_join(v_online_models_dir, v_online_algorithm.lower())
    v_online_model_file_name = path_join(v_online_model_dir, 'model.zip')
    v_online_model_file_name_stats = path_join(v_online_model_dir, 'stats.pkl')
    v_online_model_replay_buffer = path_join(v_online_model_dir, 'replay_buffer.pkl')
    v_online_model_dataset_file_name = path_join(v_online_model_dir, 'dataset.h5')

    if not os.path.exists(v_online_model_dir):
        os.makedirs(v_online_model_dir.lower())

    print("Start training model...")

    try:

        # agent = SAC(v_env,
        #             gradient_updates=20,
        #             num_q_nets=2,
        #             m_sample=None,
        #             buffer_size=int(4e5),
        #             mbpo=False,
        #             experiment_name=v_model_name,
        #             log=True,
        #             wandb=True)

        agent = SAC(v_env,
                    learning_rate=1e-5,
                    tau=0.05,
                    net_arch=[512, 512, 512, 512],
                    batch_size=512,
                    num_q_nets=4,
                    gradient_updates=20,
                    m_sample=None,
                    buffer_size=int(4e5),
                    mbpo=False,
                    experiment_name=v_model_name,
                    log=True,
                    wandb=True)

        agent.learn(total_timesteps=v_total_timesteps)
        agent.save()

    except Exception as e:
        print(e)

    print("End training online model...")

    v_env.close()


def main():
    CONFIG_FILE = 'settings/config-oanda.ini'

    v_algorithms, v_number_of_trials, v_subdir, v_train_test_split, v_train_look_back_period, v_total_timesteps, v_market, v_resolution, v_num_of_instruments, v_spread, v_num_cpu = get_train_params(
        CONFIG_FILE)

    for i in range(v_number_of_trials):
        run_trial(v_algorithms, v_train_look_back_period, v_total_timesteps, v_market, v_resolution,
                  v_num_of_instruments, v_spread, v_num_cpu, v_subdir, v_train_test_split, CONFIG_FILE)


if __name__ == "__main__":
    main()
