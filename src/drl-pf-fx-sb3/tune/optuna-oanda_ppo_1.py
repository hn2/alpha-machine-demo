#   https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py

""" Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using PPO, A2C, TD3, SAC implementation from Stable-Baselines3
on a OpenAI Gym environment.

This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.

You can run this example as follows:
    $ python sb3_simple.py

1. Increase pagefile size to 32-64 gigs
2. Use memory optimize for off-policy algos: ddpg, td3, sacSS
"""
from datetime import datetime

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.monitor import Monitor

from findrl.config_utils import get_optuna_params
from findrl.data_utils import prepare_data_optuna
from findrl.datetime_utils import WEEKDAY, get_week_number
from findrl.env_utils import make_env_pf_fx
from findrl.forex_utils import get_forex_6, get_forex_7, get_forex_10, get_forex_12, get_forex_14, get_forex_18, \
    get_forex_28, get_forex_33
from findrl.general_utils import convert_dict_to_list
from findrl.model_utils import get_online_class_sb3
from findrl.optuna_sample_params_ppo_1 import HYPERPARAMS_SAMPLER
from findrl.optuna_utils import TrialEvalCallback

# from stable_baselines3 import PPO, A2C, TD3, SAC, DDPG

CONFIG_FILE = '../settings/config-oanda-daily-on-policy.ini'
#   ALGOS = ['PPO', 'A2C', 'SAC', 'DDPG', 'TD3']
ALGOS = ['PPO']
N_TRIALS = 1200
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(5e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3
N_JOBS = 12
#   STUDY_NAME_PREFIX = 'fx1-leverage-30'
STUDY_NAME_PREFIX = 'fx-7-relu-leverage-30-5000'
NUM_OF_INSTRUMENTS = 7
RESOLUTION = 'daily'
CONNECTION_URL_LOCAL = 'mysql+pymysql://hannan:dD33dD33@localhost:3306/optuna'
CONNECTION_URL_REMOTE = 'mysql+pymysql://hannan:dD33dD33@optuna.cluster-cu3liabuijge.us-east-1.rds.amazonaws.com:3306/optuna'
MODE = 'Local'  # Local Remote

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    # "env": ENV_ID,
}


def get_envs():
    v_num_of_instruments, v_spread, v_data_dir, v_market, v_subdir, v_resolution, v_train_look_back_period, v_train_test_split, v_connection_url = get_optuna_params(
        CONFIG_FILE)

    # prepare data
    if v_num_of_instruments == 4:
        v_instruments, v_pip_size, v_pip_spread = get_forex_7(v_spread)
    elif v_num_of_instruments == 6:
        v_instruments, v_pip_size, v_pip_spread = get_forex_6(v_spread)
    elif v_num_of_instruments == 7:
        v_instruments, v_pip_size, v_pip_spread = get_forex_7(v_spread)
    elif v_num_of_instruments == 10:
        v_instruments, v_pip_size, v_pip_spread = get_forex_10(v_spread)
    elif v_num_of_instruments == 12:
        v_instruments, v_pip_size, v_pip_spread = get_forex_12(v_spread)
    elif v_num_of_instruments == 14:
        v_instruments, v_pip_size, v_pip_spread = get_forex_14(v_spread)
    elif v_num_of_instruments == 18:
        v_instruments, v_pip_size, v_pip_spread = get_forex_18(v_spread)
    elif v_num_of_instruments == 28:
        v_instruments, v_pip_size, v_pip_spread = get_forex_28(v_spread)
    elif v_num_of_instruments == 33:
        v_instruments, v_pip_size, v_pip_spread = get_forex_33(v_spread)

    print(f'v_train_test_split: {v_train_test_split}')

    v_data_train, v_data_eval = prepare_data_optuna(v_data_dir, v_market, v_resolution, v_instruments,
                                                    v_train_look_back_period, v_train_test_split, False)

    v_env_lookback_period, v_random_episode_start, v_cash, v_max_slippage_percent, v_lot_size, v_leverage, \
    v_compute_position, v_compute_indicators, v_compute_reward, v_meta_rl, v_env_verbose = get_pf_fx_env_params(
        CONFIG_FILE)
    v_main_dir, v_models_dir, v_logs_dir, v_data_dir = get_paths_params(CONFIG_FILE)

    v_env_train = make_env_pf_fx(v_data_train,
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

    v_env_eval = make_env_pf_fx(v_data_eval,
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

    # v_monitor = path_join(v_logs_dir, 'monitor_' + uuid.uuid4().hex)[:8] + '.csv')
    # v_env_train = Monitor(v_env_train, v_monitor)
    v_env_train = Monitor(v_env_train, v_logs_dir)
    v_env_eval = Monitor(v_env_eval, v_logs_dir)

    return v_env_train, v_env_eval


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    v_env_train, v_env_eval = get_envs()
    kwargs.update({"env": v_env_train})
    algo = study.user_attrs["algo"]
    kwargs.update(HYPERPARAMS_SAMPLER[algo.upper()](trial))
    model = get_online_class_sb3(algo)(**kwargs)
    eval_env = v_env_eval
    # Create the callback that will periodically evaluate
    # and report the performance
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":

    # Set pytorch num threads to 1 for faster training
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)
    #   pruner = SuccessiveHalvingPruner()

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

    #   optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    today = datetime.today()
    week = get_week_number(WEEKDAY.FRI, today)
    year = today.year

    for algo in ALGOS:

        # print(f'Algo: {algo}')

        STUDY_NAME = f'{STUDY_NAME_PREFIX}-{NUM_OF_INSTRUMENTS}-{algo.lower()}-{RESOLUTION}-{week}-{year}'

        study = optuna.create_study(study_name=STUDY_NAME, storage=storage, load_if_exists=True, sampler=sampler,
                                    pruner=pruner, direction="maximize")

        study.set_user_attr("algo", algo)
        print(f'Algo: {study.user_attrs["algo"]}')

        try:
            #   study.optimize(objective, n_trials=N_TRIALS, timeout=600)
            study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))
        print(f"Best trial for {algo}:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        print(convert_dict_to_list(trial.params))

        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        print("  User attrs:")
        for key, value in trial.user_attrs.items():
            print("    {}: {}".format(key, value))
