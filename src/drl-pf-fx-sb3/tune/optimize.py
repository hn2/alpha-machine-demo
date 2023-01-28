#   https://towardsdatascience.com/optuna-vs-hyperopt-which-hyperparameter-optimization-library-should-you-choose-ed8564618151
#   https://stable-baselines.readthedocs.io/en/master/modules/sac.html

#   https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/best-practices-ppo.md
#   https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe

#   pip install mysql-connector-python

import json
import uuid
from datetime import datetime, timedelta
from os.path import join as path_join
from pprint import pprint

import gym
import h5py
import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.sac.policies import MlpPolicy as SacMlpPolicy
from stable_baselines3.td3.policies import MlpPolicy as Td3MlpPolicy
from utils_lib.my_forex_instruments import my_forex_instruments
from utils_lib.my_globals import DATASETS_DIR, OPTUNA_LOGS_DIR

#   python -m tensorboard.main --logdir=.
#   tensorboard --port 7000 --logdir=.
#   tensorboard --logdir=.

DELIMITER = '-'
VERBOSE = False
MODEL_NAME = None
CREATE_TENSORBOARD_LOG = False
MARKET = 'oanda'  # oanda   fxcm
RESOLUTION = 'daily'  # day  hour
COMPUTE_REWARD = ['log_returns']
ONLINE_ALGORITHM = 'SAC'  # SAC TD3
# NUMBER_OF_INSTRUMENTS = 7
# TRAIN_LOOKBACK_PERIOD = 2000
#   STUDY_NAME = "'fx-7-td3-log_returns-oanda-day-12-2-2021'"
#   STUDY_NAME = "'fx-7-td3-log_returns-oanda-day-14-2-2021'"
STUDY_NAME = None
NUMBER_OF_TRIALS = 500  # ~ 6 hours for 100 trials using 12 jobs and current paramaters
N_JOBS = 6

TRAIN_TEST_ALL = 'train'
LOOKBACK = 2000
TRAIN_TEST_SPLIT = 0.8


class optimise():

    def __init__(self):

        self.name = None

    def get_dataset_name(self, instruments, market, resolution, train_lookback_period):

        dataset__name = 'fx-' + str(len(instruments)) + DELIMITER + market + DELIMITER + resolution + \
                        DELIMITER + TRAIN_TEST_ALL + DELIMITER + str(
            train_lookback_period) + '.h5'

        print(dataset__name)

        return dataset__name

    def get_env(self, env_params):

        v_spread = env_params['spread']
        v_number_of_instruments = env_params['number_of_instruments']

        if v_number_of_instruments == 4:
            v_instruments, v_pip_size, v_pip_spread = my_forex_instruments.get_forex_4(v_spread)
        elif v_number_of_instruments == 7:
            v_instruments, v_pip_size, v_pip_spread = my_forex_instruments.get_forex_7(v_spread)
        elif v_number_of_instruments == 12:
            v_instruments, v_pip_size, v_pip_spread = my_forex_instruments.get_forex_12(v_spread)
        elif v_number_of_instruments == 18:
            v_instruments, v_pip_size, v_pip_spread = my_forex_instruments.get_forex_18(v_spread)
        elif v_number_of_instruments == 28:
            v_instruments, v_pip_size, v_pip_spread = my_forex_instruments.get_forex_28(v_spread)

        v_max_slippage_percent = 1e-2

        print(f'Instruments: {v_instruments}')

        v_lot_size = 'Standard'
        v_leverage = 2e1

        v_cash = 1e3

        v_use_spread = True

        v_market = MARKET
        v_resolution = RESOLUTION

        # v_train_lookback_period = TRAIN_LOOKBACK_PERIOD
        v_train_lookback_period = int(LOOKBACK * TRAIN_TEST_SPLIT)

        '''
        if v_resolution == 'hour':
            v_env_lookback_period = 24
        elif v_RESOLUTION == 'daily':
            v_env_lookback_period = 20
        '''
        v_env_lookback_period = env_params['lookback']

        v_compute_position = 'long_and_short'  # long_only   long_and_short   short_only
        v_compute_indicators = env_params[
            'compute_indicators']  # none returns ptr returns_hlc default all all_hlc all_multi atr rsi misc ptr_returns momentum momentum_hlc momentum_multi momentum_ptr_returns
        v_compute_reward = COMPUTE_REWARD
        v_verbose = VERBOSE

        v_dataset_name = self.get_dataset_name(v_instruments, v_market, v_resolution, v_train_lookback_period)

        v_dataset_file_name = path_join(DATASETS_DIR, v_dataset_name)
        print(f'Using {v_dataset_file_name}')

        with h5py.File(v_dataset_file_name, 'r') as f:
            v_data = f['dataset_1'][:]

        v_env = gym.make('gym_fx:FxEnv-v0',
                         data=v_data,
                         instruments=v_instruments,
                         lookback=v_env_lookback_period,
                         random_episode_start=True,
                         cash=v_cash,
                         max_slippage_percent=v_max_slippage_percent,
                         lot_size=v_lot_size,
                         leverage=v_leverage,
                         pip_size=v_pip_size,
                         pip_spread=v_pip_spread,
                         compute_position=v_compute_position,
                         compute_indicators=v_compute_indicators,
                         compute_reward=v_compute_reward,  # returns log_returns
                         verbose=v_verbose)

        return v_env

    def optimize_env(self, trial):
        """ Learning hyperparamters we want to optimise"""
        return {
            'number_of_instruments': trial.suggest_categorical('number_of_instruments', [7, 12, 18]),
            'spread': trial.suggest_discrete_uniform('spread', 2, 5, 1),
            # 'lookback': trial.suggest_int('lookback', 15, 25, 1),
            'lookback': trial.suggest_int('lookback', 30, 50, 5),
            #   'compute_indicators': trial.suggest_categorical("compute_indicators", ['all'])
            'compute_indicators': trial.suggest_categorical("compute_indicators", ['all'])
            #   'compute_indicators':  trial.suggest_categorical("compute_indicators", ['prices', 'returns', 'log_returns', 'returns_hlc', 'log_returns_hlc', 'patterns', 'returns_patterns_volatility', 'momentum', 'all', 'misc']),
        }

    def optimize_model(self, trial):

        """ Learning hyperparamters we want to optimise"""

        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)  # 1e-5
        #   learning_rate = trial.suggest_loguniform("learning_rate", 5e-6, 1)     #    1e-5
        #   learning_rate = trial.suggest_loguniform("learning_rate", 5e-6, 5e-4)  # 1e-5
        #   learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 5e-4)  # 1e-5
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048])
        #   batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
        buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
        #   learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 10000, 20000])
        learning_starts = trial.suggest_categorical("learning_starts", [0, 10, 100, 200])
        # train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
        train_freq = trial.suggest_categorical("train_freq", [8, 16, 32, 64, 128, 256, 512])
        # Polyak coeff
        tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05])
        # gradient_steps takes too much time
        # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
        gradient_steps = train_freq
        # ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
        ent_coef = "auto"
        # You can comment that out when not using gSDE
        #   log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
        net_arch = trial.suggest_categorical('net_arch', ['small', 'medium', 'large', 'xlarge'])
        #   net_arch = trial.suggest_categorical("net_arch", ['medium', 'large', 'xlarge'])
        #   net_arch = trial.suggest_categorical('net_arch', ['xsmall', 'small', 'medium', 'large', 'xlarge'])
        #   activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])
        #   use_sde = trial.suggest_discrete_uniform("use_sde", 0, 1, 1)

        '''
        net_arch = {
            "large": [256, 256],
            "xlarge": [512, 512],
            "xxlarge": [1024, 1024]
        }[net_arch]
        '''

        net_arch = {
            "default": [64, 64],
            "xxxsmall": [16, 16, 16, 16],
            "xxsmall": [32, 32, 32, 32],
            "xsmall": [64, 64, 64, 64],
            "small": [128, 128, 128, 128],
            "medium": [256, 256, 256, 256],
            "large": [512, 512, 512, 512],
            "xlarge": [1024, 1024, 1024, 1024],
            "xxlarge": [2048, 2048, 2048, 2048],
            "xxxlarge": [4096, 4096, 4096, 4096],
        }[net_arch]

        target_entropy = "auto"
        # if ent_coef == 'auto':
        #     # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
        #     target_entropy = trial.suggest_uniform('target_entropy', -10, 10)

        return {
            #   "policy": "MlpPolicy",
            "gamma": gamma,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "learning_starts": learning_starts,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            #   "ent_coef": ent_coef,
            "tau": tau,
            #   "target_entropy": target_entropy,
            #   "policy_kwargs": dict(log_std_init=log_std_init, net_arch=net_arch),
            "policy_kwargs": dict(net_arch=net_arch),
            #   "use_sde": bool(use_sde)
        }

    def optimize_noise(self, trial):

        """ Learning hyperparamters we want to optimise"""

        return {
            #   'total_timestamp':  trial.suggest_discrete_uniform('total_timestamp', 10000, 50000, 5000)
            'action_noise': trial.suggest_categorical("action_noise",
                                                      ['None', 'NormalActionNoise', 'OrnsteinUhlenbeckActionNoise'])
            #   'action_noise': trial.suggest_categorical("action_noise", ['OrnsteinUhlenbeckActionNoise'])
        }

    def optimize_learn(self, trial):

        """ Learning hyperparamters we want to optimise"""

        return {
            'total_timestamp': trial.suggest_discrete_uniform('total_timestamp', 20000, 20000, 20000)
        }

    def get_online_class_and_policy(self, algorithm):

        if algorithm == 'SAC':
            online_class = SAC
            online_policy = SacMlpPolicy
        elif algorithm == 'TD3':
            online_class = TD3
            online_policy = Td3MlpPolicy

        return online_class, online_policy

    def get_algo_name(self, algo):

        if algo == SAC:
            return 'SAC'
        elif algo == 'TD3':
            return 'TD3'

    def optimize_agent(self, trial):
        """ Train the model and optimise
            Optuna maximises the negative log likelihood, so we
            need to negate the reward here
        """
        env_params = self.optimize_env(trial)
        model_params = self.optimize_model(trial)
        noise_paramas = self.optimize_noise(trial)
        learn_params = self.optimize_learn(trial)

        pprint(env_params, width=1)
        pprint(model_params, width=1)
        pprint(noise_paramas, width=1)
        pprint(learn_params, width=1)

        v_env = self.get_env(env_params)

        if CREATE_TENSORBOARD_LOG:
            v_monitor = path_join(OPTUNA_LOGS_DIR, MODEL_NAME)
            v_dummy_vec_env = DummyVecEnv([lambda: Monitor(v_env, v_monitor)])
            v_dummy_vec_env.seed(1)
            v_vec_normalize = VecNormalize(venv=v_dummy_vec_env, training=True, norm_obs=True, norm_reward=True,
                                           clip_obs=10.0, clip_reward=10.0, gamma=model_params['gamma'], epsilon=1e-08)
            v_vec_normalize.seed(1)

            online_class, online_policy = self.get_online_class_and_policy(ONLINE_ALGORITHM)

            n_actions = v_dummy_vec_env.action_space.shape[-1]

            if noise_paramas['action_noise'] == 'None':
                v_action_noise = None
            if noise_paramas['action_noise'] == 'NormalActionNoise':
                v_action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            if noise_paramas['action_noise'] == 'OrnsteinUhlenbeckActionNoise':
                v_action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

            online_model = online_class(env=v_vec_normalize, policy=online_policy, verbose=VERBOSE,
                                        action_noise=v_action_noise, optimize_memory_usage=True, **model_params,
                                        tensorboard_log=OPTUNA_LOGS_DIR)

        else:
            v_dummy_vec_env = DummyVecEnv([lambda: v_env])
            v_dummy_vec_env.seed(1)
            v_vec_normalize = VecNormalize(venv=v_dummy_vec_env, training=True, norm_obs=True, norm_reward=True,
                                           clip_obs=10.0, clip_reward=10.0, gamma=model_params['gamma'], epsilon=1e-08)
            v_vec_normalize.seed(1)

            online_class, online_policy = self.get_online_class_and_policy(ONLINE_ALGORITHM)

            n_actions = v_dummy_vec_env.action_space.shape[-1]

            if noise_paramas['action_noise'] == 'None':
                v_action_noise = None
            if noise_paramas['action_noise'] == 'NormalActionNoise':
                v_action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            if noise_paramas['action_noise'] == 'OrnsteinUhlenbeckActionNoise':
                v_action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

            online_model = online_class(env=v_vec_normalize, policy=online_policy, verbose=VERBOSE,
                                        action_noise=v_action_noise, optimize_memory_usage=True, **model_params,
                                        tensorboard_log=OPTUNA_LOGS_DIR)

        online_model.learn(total_timesteps=(int(learn_params['total_timestamp'])))

        rewards = []
        n_episodes, reward_sum = 0, 0.0

        obs = v_env.reset()

        while n_episodes < 4:
            action, _ = online_model.predict(obs)
            obs, reward, done, _ = v_env.step(action)
            reward_sum += reward

            if done:
                rewards.append(reward_sum)
                reward_sum = 0.0
                n_episodes += 1
                obs = v_env.reset()

        mean_reward = np.mean(rewards)
        '''
        trial.report(-1 * mean_reward, 1)
    
        return -1 * mean_reward
        '''
        trial.report(mean_reward, 1)

        return mean_reward


if __name__ == '__main__':
    # print command line arguments
    '''
    for arg in sys.argv[1:]:
        print arg
    '''

    my_optimise = optimise()

    #   market = 'oanda'   #   'oanda' - 'fxcm'
    #   RESOLUTION = 'daily'   #   'day' - 'hour'

    now = datetime.now() + timedelta(days=0)

    v_uuid = (uuid.uuid4().hex)[:8]

    #   MODEL_NAME = 'fx-' + str(NUMBER_OF_INSTRUMENTS) + DELIMITER + ONLINE_ALGORITHM.lower() + DELIMITER + MARKET + DELIMITER + RESOLUTION + DELIMITER + str(now.day-1) + DELIMITER + str(now.month) + DELIMITER + str(now.year) + DELIMITER + v_uuid
    # STUDY_NAME = 'fx-' + str(
    #     NUMBER_OF_INSTRUMENTS) + DELIMITER + ONLINE_ALGORITHM.lower() + DELIMITER + MARKET + DELIMITER + RESOLUTION + DELIMITER + str(
    #     now.day) + DELIMITER + str(now.month) + DELIMITER + str(now.year)

    if STUDY_NAME is None:
        # STUDY_NAME = f'fx-{NUMBER_OF_INSTRUMENTS}-{ONLINE_ALGORITHM.lower()}-{"-".join(COMPUTE_REWARD)}-{MARKET}-{RESOLUTION}-{now.day}-{now.month}-{now.year}'
        STUDY_NAME = f'fx-{ONLINE_ALGORITHM.lower()}-{"-".join(COMPUTE_REWARD)}-{MARKET}-{RESOLUTION}-{now.day}-{now.month}-{now.year}'

    v_study_params = path_join(OPTUNA_LOGS_DIR, STUDY_NAME + '_params.json')

    storage = optuna.storages.RDBStorage(
        #   url='postgresql://hannan:dD33dD33@database-1.cu3liabuijge.us-east-1.rds.amazonaws.com:5432/optuna', #   aws
        #   url='postgresql://postgres:dD33dD33@localhost:5432/optuna',  # local postgress
        #   DATABSE_URI='mysql+mysqlconnector://{user}:{password}@{server}/{database}'.format(user='your_user', password='password', server='localhost', database='dname')
        url='mysql+mysqlconnector://hannan:dD33dD33@localhost:3306/optuna',
        engine_kwargs={
            'pool_size': 20,
            'max_overflow': 0
        }
    )

    '''
    storage = RedisStorage(
            url="redis://passwd@localhost:port/db",
        )
    '''
    #   study = optuna.create_study(study_name=MODEL_NAME, storage=storage, load_if_exists=True, pruner=optuna.pruners.MedianPruner())
    #   study = optuna.create_study(study_name=MODEL_NAME, storage='sqlite:///params.db', load_if_exists=True, pruner=optuna.pruners.MedianPruner())

    print(f'STUDY_NAME={STUDY_NAME}')

    study = optuna.create_study(study_name=STUDY_NAME, storage=storage, load_if_exists=True, direction='maximize',
                                sampler=TPESampler(), pruner=SuccessiveHalvingPruner())

    study.optimize(my_optimise.optimize_agent, n_trials=NUMBER_OF_TRIALS, n_jobs=N_JOBS)  # n_jobs=-1

    with open(v_study_params, 'w') as f:
        json.dump(dict(best_params=study.best_params, best_value=study.best_value), f, indent=2)
