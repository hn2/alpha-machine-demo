#   https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
#   https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
#   https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
#   https://github.com/araffin/rl-baselines-zoo

import os
import random
from datetime import datetime
from os.path import join as path_join

import numpy as np
import optuna
from d3rlpy.wrappers.sb3 import to_mdp_dataset
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

#   sys.path.append("../findrl")  # Adds higher directory to python modules path.
from findrl.config_utils import get_paths_params, get_pf_fx_env_params, get_train_params, \
    get_agent_params, get_model_params
from findrl.data_utils import prepare_data_train, calculate_features
from findrl.datetime_utils import WEEKDAY, get_week_number
from findrl.env_utils import make_env_pf_fx
from findrl.forex_utils import get_forex_7, get_forex_12, get_forex_14, get_forex_18, get_forex_28
from findrl.model_utils import get_model_name, get_online_class_and_policy
from findrl.optuna_best_params_10 import HYPERPARAMS_BEST

CONFIG_FILE = 'settings/config-oanda-train-test-daily.ini'
USE_STUDY_NAME = None
STUDY_NAME_PREFIX = 'fx1-leverage-20-1000'
NUM_OF_INSTRUMENTS = 7
RESOLUTION = 'daily'
CONNECTION_URL_LOCAL = 'mysql+pymysql://hannan:dD33dD33@localhost:3306/optuna'
CONNECTION_URL_REMOTE = 'mysql+pymysql://hannan:dD33dD33@optuna.cluster-cu3liabuijge.us-east-1.rds.amazonaws.com:3306/optuna'
MODE = 'Remote'  # Local Remote

DEFAULT_HYPERPARAMS = {
    # "policy": "MlpPolicy",
    # "env": ENV_ID,
}


def prepare_env(online_algorithm, train_look_back_period, total_timesteps, market, resolution, num_of_instruments,
                spread, num_cpu, subdir, train_test_split, config_file):
    v_main_dir, v_models_dir, v_logs_dir, v_data_dir = get_paths_params(config_file)
    v_delimeter, v_model_prefix = get_model_params(config_file)
    v_model_verbose, v_callback_verbose, v_save_replay_buffer, v_use_tensorboard, v_use_callback, v_check_freq, \
    v_callback_lookback, v_save_freq, v_a2c_params, v_ppo_params, v_ddpg_params, v_dqn_params, v_sac_params, v_td3_params, v_net_arch, v_use_linear_schedule, v_action_noise, v_noise_sigma, v_use_sde = get_agent_params(
        config_file)
    v_env_lookback_period, v_random_episode_start, v_cash, v_max_slippage_percent, v_lot_size, v_leverage, \
    v_compute_position, v_compute_indicators, v_compute_reward, v_meta_rl, v_env_verbose = get_pf_fx_env_params(
        config_file)

    v_online_algorithm = random.choice(online_algorithm)
    v_num_of_instruments = int(random.choice(num_of_instruments))

    today = datetime.today()
    WEEK = get_week_number(WEEKDAY.FRI, today)
    YEAR = today.year

    v_total_timesteps = total_timesteps + WEEK

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

    v_features = calculate_features(v_data, v_compute_indicators)
    # v_data = prepare_data_train(v_market, v_resolution, v_instruments, v_train_look_back_period)

    print(f'Data shape:{np.shape(v_data)}, Features shape:{np.shape(v_features)}')

    #   v_compute_position = random.choice(v_compute_position)

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

    v_eval_env = make_env_pf_fx(v_data,
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

    print(
        f'Instruments:{v_instruments}, lookack:{v_env_lookback_period}, random_episode_start:{v_random_episode_start}, cash:{v_cash}, max_slippage_percent:{v_max_slippage_percent}, lot_size:{v_lot_size}, leverage:{v_leverage}, pip_size:{v_pip_size}, pip_spread:{v_pip_spread}, compute_position:{v_compute_position}, compute_indicators:{v_compute_indicators}, compute_reward:{v_compute_reward}, meta_rl:{v_meta_rl}, verbose:{v_env_verbose}')

    return v_env, v_eval_env, v_instruments


def run_trial(env, eval_env, instruments, online_algorithm, train_look_back_period, total_timesteps, market, resolution,
              num_of_instruments, spread,
              num_cpu, subdir, train_test_split, config_file):
    v_main_dir, v_models_dir, v_logs_dir, v_data_dir = get_paths_params(config_file)
    v_delimeter, v_model_prefix = get_model_params(config_file)
    v_model_verbose, v_callback_verbose, v_save_replay_buffer, v_use_tensorboard, v_use_callback, v_check_freq, \
    v_callback_lookback, v_save_freq, v_a2c_params, v_ppo_params, v_ddpg_params, v_dqn_params, v_sac_params, v_td3_params, v_net_arch, v_use_linear_schedule, v_action_noise, v_noise_sigma, v_use_sde = get_agent_params(
        config_file)
    v_env_lookback_period, v_random_episode_start, v_cash, v_max_slippage_percent, v_lot_size, v_leverage, \
    v_compute_position, v_compute_indicators, v_compute_reward, v_meta_rl, v_env_verbose = get_pf_fx_env_params(
        config_file)

    v_online_algorithm = random.choice(online_algorithm)
    v_num_of_instruments = int(random.choice(num_of_instruments))

    today = datetime.today()
    WEEK = get_week_number(WEEKDAY.FRI, today)
    YEAR = today.year

    v_total_timesteps = total_timesteps + WEEK

    v_action_noise = random.choice(v_action_noise)

    v_model_name = get_model_name(v_delimeter, v_model_prefix, v_leverage, v_action_noise, v_use_callback,
                                  v_random_episode_start, instruments, v_total_timesteps, train_look_back_period,
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

    v_monitor = path_join(v_logs_dir, v_model_name)
    v_dummy_vec_env = DummyVecEnv([lambda: Monitor(env, v_monitor)])
    v_dummy_vec_env.seed(1)

    online_class, online_policy = get_online_class_and_policy(v_online_algorithm)

    # n_actions = v_dummy_vec_env.action_space.shape[-1]
    # v_action_noise_class = get_action_noise_class(v_online_algorithm, v_action_noise, n_actions, v_noise_sigma)

    # load recent checkpoint
    if os.path.isfile(v_online_model_file_name) and os.path.isfile(v_online_model_file_name_stats):
        v_vec_normalize = VecNormalize.load(v_online_model_file_name_stats, v_dummy_vec_env)
        v_vec_normalize.reset()
        v_online_model = online_class.load(v_online_model_file_name, v_vec_normalize)
        print('Model Loaded ...')
    else:
        #   v_vec_normalize = VecNormalize(v_dummy_vec_env, norm_obs, norm_reward, clip_obs, clip_reward, gamma)
        v_vec_normalize = VecNormalize(v_dummy_vec_env)
        v_vec_normalize.seed(1)

    print(f'Mode: {MODE}')

    storage = optuna.storages.RDBStorage(
        #   url='postgresql://hannan:dD33dD33@database-1.cu3liabuijge.us-east-1.rds.amazonaws.com:5432/optuna', #   aws
        #   url='postgresql://postgres:dD33dD33@localhost:5432/optuna',  # local postgress
        #   DATABSE_URI='mysql+mysqlconnector://{user}:{password}@{server}/{database}'.format(user='your_user', password='password', server='localhost', database='dname')
        # url='mysql+mysqlconnector://hannan:dD33dD33@localhost:3306/optuna',
        url=CONNECTION_URL_LOCAL if MODE.lower() == 'local' else CONNECTION_URL_REMOTE,
        engine_kwargs={
            'pool_size': 20,
            'max_overflow': 0
        }
    )

    if USE_STUDY_NAME is None:
        STUDY_NAME = f'{STUDY_NAME_PREFIX}-{NUM_OF_INSTRUMENTS}-{v_online_algorithm.lower()}-{RESOLUTION}-{WEEK}-{YEAR}'
    else:
        STUDY_NAME = USE_STUDY_NAME

    print(f'Study name: {STUDY_NAME}')

    study = optuna.create_study(study_name=STUDY_NAME, storage=storage, load_if_exists=True, direction="maximize")
    best_trial = study.best_trial

    print("Number of finished trials: ", len(study.trials))

    kwargs = DEFAULT_HYPERPARAMS.copy()

    kwargs.update(HYPERPARAMS_BEST[v_online_algorithm.upper()](best_trial))

    print(f'Algo: {v_online_algorithm}')
    print(f"Best trial value for {v_online_algorithm}: {best_trial.value}")
    print(kwargs)

    # if v_online_algorithm == 'PPO' or v_online_algorithm == 'A2C' or v_online_algorithm == 'SAC':
    #     v_online_model = online_class(env=v_vec_normalize, policy=online_policy,
    #                                   verbose=v_model_verbose, **kwargs, use_sde=v_use_sde,
    #                                   tensorboard_log=v_logs_dir if v_use_tensorboard else None)
    # elif v_online_algorithm == 'TD3':
    #     v_online_model = online_class(env=v_vec_normalize, policy=online_policy,
    #                                   action_noise=v_action_noise_class, optimize_memory_usage=True,
    #                                   verbose=v_model_verbose, **kwargs,
    #                                   tensorboard_log=v_logs_dir if v_use_tensorboard else None)

    v_online_model = online_class(env=v_vec_normalize, policy=online_policy,
                                  verbose=v_model_verbose, **kwargs,
                                  tensorboard_log=v_logs_dir if v_use_tensorboard else None)

    # replay buffer
    if os.path.isfile(v_online_model_replay_buffer):
        v_online_model.load_replay_buffer(v_online_model_replay_buffer)

    print("Start training model...")

    try:

        if v_use_callback:
            # callback = SaveToDiskOnBestTrainingRewardCallback(check_freq=v_check_freq, save_freq=v_save_freq,
            #                                                   lookback=v_callback_lookback,
            #                                                   online_algorithm=v_online_algorithm,
            #                                                   model_file_name=v_online_model_file_name,
            #                                                   model_replay_buffer=v_online_model_replay_buffer,
            #                                                   model_stats=v_online_model_file_name_stats,
            #                                                   save_replay_buffer=v_save_replay_buffer,
            #                                                   verbose=v_callback_verbose)

            eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                         log_path='./logs/', eval_freq=500,
                                         deterministic=True, render=False)

            v_online_model.learn(total_timesteps=v_total_timesteps, log_interval=1000, reset_num_timesteps=False,
                                 tb_log_name=v_model_name, callback=eval_callback)
        else:
            v_online_model.learn(total_timesteps=v_total_timesteps, log_interval=1000, reset_num_timesteps=False,
                                 tb_log_name=v_model_name)

        if not v_use_callback:
            v_online_model.save(v_online_model_file_name.lower())
            v_vec_normalize.save(v_online_model_file_name_stats.lower())

    except Exception as e:
        print(e)

    if v_save_replay_buffer:
        try:
            dataset = to_mdp_dataset(v_online_model.replay_buffer)
            dataset.dump(v_online_model_dataset_file_name)
            os.remove(v_online_model_replay_buffer.lower())
            # os.remove(v_online_model_file_name)
            # os.remove(v_online_model_file_name_stats)
        except Exception as e:
            print(e)

    print("End training online model...")

    env.close()


def main():
    v_algorithms, v_number_of_trials, v_subdir, v_train_test_split, v_train_look_back_period, v_total_timesteps, v_market, v_resolution, v_num_of_instruments, v_spread, v_num_cpu = get_train_params(
        CONFIG_FILE)

    v_env, v_eval_env, v_instruments = prepare_env(v_algorithms, v_train_look_back_period, v_total_timesteps, v_market,
                                                   v_resolution,
                                                   v_num_of_instruments, v_spread, v_num_cpu, v_subdir,
                                                   v_train_test_split,
                                                   CONFIG_FILE)

    for i in range(v_number_of_trials):
        run_trial(v_env, v_env, v_instruments, v_algorithms, v_train_look_back_period, v_total_timesteps, v_market,
                  v_resolution, v_num_of_instruments, v_spread, v_num_cpu, v_subdir, v_train_test_split, CONFIG_FILE)


if __name__ == "__main__":
    main()
