#   https://stackoverflow.com/questions/43457222/run-python-subprocess-using-popen-independently
import os
import os.path
import platform
import subprocess
from os.path import join as path_join

import psutil
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .env_utils import make_env_pf_fx, make_env_pf_fx_qc, make_env_choose_best_model
from .file_utils import dump_dict_to_pickle, replace_in_file
from .model_utils import parse_model_name_online_algorithm, get_online_class_sb3, \
    load_model_from_json_d3rlpy


def run_model_make_env_sb3(models_dir,
                           model_name,
                           delimeter,
                           deterministic,
                           data,
                           account_currency,
                           instruments,
                           env_lookback_period,
                           random_episode_start,
                           cash,
                           max_slippage_percent,
                           lot_size,
                           leverage,
                           pip_size,
                           pip_spread,
                           compute_position,
                           compute_indicators,
                           compute_reward,  # returns log_returns
                           meta_rl,
                           env_verbose):
    v_env = make_env_pf_fx(data,
                           account_currency,
                           instruments,
                           env_lookback_period,
                           random_episode_start,
                           cash,
                           max_slippage_percent,
                           lot_size,
                           leverage,
                           pip_size,
                           pip_spread,
                           compute_position,
                           compute_indicators,
                           compute_reward,  # returns log_returns
                           meta_rl,
                           env_verbose)

    #   check_env(v_env)

    v_online_model_dir = path_join(*[models_dir, model_name, 'online'])

    #   check if exists env.pkl
    # v_online_model_env_file_name = path_join(v_online_model_dir, 'env.pkl')
    # if not os.path.exists(v_online_model_env_file_name):
    #     v_env_attributes = dict(v_env.__dict__)
    #     v_env_attributes.pop('data')
    #     v_env_attributes.pop('price_data')
    #     v_env_attributes.pop('features_data')
    #     v_env_attributes['account_currency'] = 'usd'
    #     dump_dict_to_pickle(v_online_model_env_file_name, v_env_attributes)
    #
    #     v_online_model_instruments_file_name = path_join(v_online_model_dir, 'instruments.pkl')
    #     if os.path.exists(v_online_model_instruments_file_name):
    #         os.remove(v_online_model_instruments_file_name)

    v_online_model_file_name = path_join(v_online_model_dir, 'model')
    v_online_model_file_name_stats = path_join(v_online_model_dir, 'stats.pkl')

    v_dummy_vec_env = DummyVecEnv([lambda: v_env])
    v_vec_normalize = VecNormalize.load(v_online_model_file_name_stats, v_dummy_vec_env)
    v_vec_normalize.training = False
    v_vec_normalize.norm_reward = False

    online_algorithm = parse_model_name_online_algorithm(model_name, delimeter)

    online_class = get_online_class_sb3(online_algorithm)
    online_model = online_class.load(v_online_model_file_name)

    online_model_parama = online_model.get_parameters()

    obs = v_vec_normalize.reset()
    done = False

    while not done:
        action, _states = online_model.predict(obs, deterministic=deterministic)
        obs, _, done, _ = v_vec_normalize.step(action)
        suggested_weights = v_vec_normalize.get_attr('last_suggested_weights')[0]
        suggested_positions = v_vec_normalize.get_attr('last_suggested_positions')[0]

    return suggested_weights, suggested_positions


def run_model_make_env_sb3_qc(models_dir,
                              model_name,
                              delimeter,
                              deterministic,
                              data,
                              start_date,
                              end_date,
                              number_of_instruments,
                              random_episode_start,
                              env_lookback_period,
                              args,
                              kwargs,
                              compute_position,
                              compute_indicators,
                              compute_reward,  # returns log_returns
                              meta_rl,
                              env_verbose):
    v_env = make_env_pf_fx_qc(data,
                              start_date,
                              end_date,
                              number_of_instruments,
                              random_episode_start,
                              env_lookback_period,
                              args,
                              kwargs,
                              compute_position,
                              compute_indicators,
                              compute_reward,  # returns log_returns
                              meta_rl,
                              env_verbose)

    #   check_env(v_env)

    v_online_model_dir = path_join(*[models_dir, model_name, 'online'])

    #   check if exists env.pkl
    # v_online_model_env_file_name = path_join(v_online_model_dir, 'env.pkl')
    # if not os.path.exists(v_online_model_env_file_name):
    #     v_env_attributes = dict(v_env.__dict__)
    #     v_env_attributes.pop('data')
    #     v_env_attributes.pop('price_data')
    #     v_env_attributes.pop('features_data')
    #     v_env_attributes['account_currency'] = 'usd'
    #     dump_dict_to_pickle(v_online_model_env_file_name, v_env_attributes)
    #
    #     v_online_model_instruments_file_name = path_join(v_online_model_dir, 'instruments.pkl')
    #     if os.path.exists(v_online_model_instruments_file_name):
    #         os.remove(v_online_model_instruments_file_name)

    v_online_model_file_name = path_join(v_online_model_dir, 'model')
    v_online_model_file_name_stats = path_join(v_online_model_dir, 'stats.pkl')

    v_dummy_vec_env = DummyVecEnv([lambda: v_env])
    v_vec_normalize = VecNormalize.load(v_online_model_file_name_stats, v_dummy_vec_env)
    v_vec_normalize.training = False
    v_vec_normalize.norm_reward = False

    online_algorithm = parse_model_name_online_algorithm(model_name, delimeter)

    online_class = get_online_class_sb3(online_algorithm)
    online_model = online_class.load(v_online_model_file_name)

    online_model_parama = online_model.get_parameters()

    obs = v_vec_normalize.reset()
    done = False

    while not done:
        action, _states = online_model.predict(obs, deterministic=deterministic)
        obs, _, done, _ = v_vec_normalize.step(action)
        suggested_weights = v_vec_normalize.get_attr('last_suggested_weights')[0]
        suggested_positions = v_vec_normalize.get_attr('last_suggested_positions')[0]

    return suggested_weights, suggested_positions


def run_model_make_env_d3rlpy(models_dir,
                              model_name,
                              delimeter,
                              deterministic,
                              data,
                              account_currency,
                              instruments,
                              env_lookback_period,
                              random_episode_start,
                              cash,
                              max_slippage_percent,
                              lot_size,
                              leverage,
                              pip_size,
                              pip_spread,
                              compute_position,
                              compute_indicators,
                              compute_reward,  # returns log_returns
                              meta_rl,
                              env_verbose):
    v_env = make_env_pf_fx(data,
                           account_currency,
                           instruments,
                           env_lookback_period,
                           random_episode_start,
                           cash,
                           max_slippage_percent,
                           lot_size,
                           leverage,
                           pip_size,
                           pip_spread,
                           compute_position,
                           compute_indicators,
                           compute_reward,  # returns log_returns
                           meta_rl,
                           env_verbose)

    #   check_env(v_env)

    v_online_model_dir = path_join(*[models_dir, model_name, 'online'])

    #   check if exists env.pkl
    v_online_model_env_file_name = path_join(v_online_model_dir, 'env.pkl')
    if not os.path.exists(v_online_model_env_file_name):
        v_env_attributes = dict(v_env.__dict__)
        v_env_attributes.pop('data')
        v_env_attributes.pop('price_data')
        v_env_attributes.pop('features_data')
        v_env_attributes['account_currency'] = 'usd'
        dump_dict_to_pickle(v_online_model_env_file_name, v_env_attributes)

        v_online_model_instruments_file_name = path_join(v_online_model_dir, 'instruments.pkl')
        if os.path.exists(v_online_model_instruments_file_name):
            os.remove(v_online_model_instruments_file_name)

    v_online_model_file_name_json = path_join(v_online_model_dir, 'params.json')
    v_online_model_file_name = path_join(v_online_model_dir, 'model.pt')

    v_online_algorithm = parse_model_name_online_algorithm(model_name, delimeter)

    print(v_online_algorithm)
    print(v_online_model_file_name_json)
    print(v_online_model_file_name)

    v_online_model = load_model_from_json_d3rlpy(v_online_algorithm, v_online_model_file_name_json,
                                                 v_online_model_file_name)

    print(v_online_model)

    obs = v_env.reset()
    done = False

    while not done:
        action = v_online_model.predict(obs)
        obs, _, done, _ = v_env.step(action)
        suggested_weights = v_env.last_suggested_weights
        suggested_positions = v_env.last_suggested_positions

    return suggested_weights, suggested_positions


def run_model_make_env_new(models_dir,
                           model_name,
                           delimeter,
                           deterministic,
                           data,
                           features,
                           instruments,
                           env_lookback_period,
                           random_episode_start,
                           cash,
                           max_slippage_percent,
                           lot_size,
                           leverage,
                           pip_size,
                           pip_spread,
                           compute_position,
                           compute_reward,  # returns log_returns
                           meta_rl,
                           env_verbose):
    v_env = make_env_pf_fx(data,
                           features,
                           instruments,
                           env_lookback_period,
                           random_episode_start,
                           cash,
                           max_slippage_percent,
                           lot_size,
                           leverage,
                           pip_size,
                           pip_spread,
                           compute_position,
                           compute_reward,  # returns log_returns
                           meta_rl,
                           env_verbose)

    #   check_env(v_env)

    v_online_model_dir = path_join(*[models_dir, model_name, 'online'])

    v_online_model_file_name = path_join(v_online_model_dir, 'model')
    v_online_model_file_name_stats = path_join(v_online_model_dir, 'stats.pkl')

    v_dummy_vec_env = DummyVecEnv([lambda: v_env])
    v_vec_normalize = VecNormalize.load(v_online_model_file_name_stats, v_dummy_vec_env)
    v_vec_normalize.training = False
    v_vec_normalize.norm_reward = False

    online_algorithm = parse_model_name_online_algorithm(model_name, delimeter)

    online_class = get_online_class(online_algorithm)
    online_model = online_class.load(v_online_model_file_name)

    online_model_parama = online_model.get_parameters()

    obs = v_vec_normalize.reset()

    done = False

    while not done:
        action, _states = online_model.predict(obs, deterministic=deterministic)
        obs, _, done, _ = v_vec_normalize.step(action)
        suggested_weights = v_vec_normalize.get_attr('last_suggested_weights')[0]
        suggested_positions = v_vec_normalize.get_attr('last_suggested_positions')[0]

    return suggested_weights, suggested_positions


def run_model_create_env(models_dir,
                         model_name,
                         delimeter,
                         deterministic,
                         data,
                         instruments,
                         env_lookback_period,
                         random_episode_start,
                         cash,
                         max_slippage_percent,
                         lot_size,
                         leverage,
                         pip_size,
                         pip_spread,
                         compute_position,
                         compute_indicators,
                         compute_reward,  # returns log_returns
                         meta_rl,
                         env_verbose):
    v_env = make_env_pf_fx(data,
                           instruments,
                           env_lookback_period,
                           random_episode_start,
                           cash,
                           max_slippage_percent,
                           lot_size,
                           leverage,
                           pip_size,
                           pip_spread,
                           compute_position,
                           compute_indicators,
                           compute_reward,  # returns log_returns
                           meta_rl,
                           env_verbose)

    #   check_env(v_env)

    v_online_model_dir = path_join(*[models_dir, model_name, 'online'])

    v_online_model_file_name = path_join(v_online_model_dir, 'model')
    v_online_model_file_name_stats = path_join(v_online_model_dir, 'stats.pkl')

    v_dummy_vec_env = DummyVecEnv([lambda: v_env])
    v_vec_normalize = VecNormalize.load(v_online_model_file_name_stats, v_dummy_vec_env)
    v_vec_normalize.training = False
    v_vec_normalize.norm_reward = False

    online_algorithm = parse_model_name_online_algorithm(model_name, delimeter)

    online_class = get_online_class_sb3(online_algorithm)
    online_model = online_class.load(v_online_model_file_name)

    online_model_parama = online_model.get_parameters()

    obs = v_vec_normalize.reset()

    done = False

    while not done:
        action, _states = online_model.predict(obs, deterministic=deterministic)
        obs, _, done, _ = v_vec_normalize.step(action)
        suggested_weights = v_vec_normalize.get_attr('last_suggested_weights')[0]
        suggested_positions = v_vec_normalize.get_attr('last_suggested_positions')[0]

    return suggested_weights, suggested_positions


def run_choose_best_model(choose_best_models_dir,
                          choose_best_model_name,
                          models_dir,
                          files_lookback_hours,
                          include_patterns,
                          delimeter,
                          deterministic,
                          data,
                          instruments,
                          env_lookback_period,
                          random_episode_start,
                          cash,
                          max_slippage_percent,
                          lot_size,
                          leverage,
                          pip_size,
                          pip_spread,
                          compute_position,
                          compute_indicators,
                          compute_reward,  # returns log_returns
                          meta_rl,
                          env_verbose):
    v_env = make_env_choose_best_model(models_dir,
                                       files_lookback_hours,
                                       include_patterns,
                                       delimeter,
                                       deterministic,
                                       data,
                                       instruments,
                                       env_lookback_period,
                                       random_episode_start,
                                       cash,
                                       max_slippage_percent,
                                       lot_size,
                                       leverage,
                                       pip_size,
                                       pip_spread,
                                       compute_position,
                                       compute_indicators,
                                       compute_reward,  # returns log_returns
                                       meta_rl,
                                       env_verbose)

    #   check_env(v_env)

    v_online_model_dir = path_join(*[choose_best_models_dir, choose_best_model_name, 'online'])

    v_online_model_file_name = path_join(v_online_model_dir, 'model')
    v_online_model_file_name_stats = path_join(v_online_model_dir, 'stats.pkl')

    v_dummy_vec_env = DummyVecEnv([lambda: v_env])
    v_vec_normalize = VecNormalize.load(v_online_model_file_name_stats, v_dummy_vec_env)
    v_vec_normalize.training = False
    v_vec_normalize.norm_reward = False

    online_algorithm = parse_model_name_online_algorithm(choose_best_model_name, delimeter)

    online_class = get_online_class_sb3(online_algorithm)
    online_model = online_class.load(v_online_model_file_name)

    online_model_parama = online_model.get_parameters()

    obs = v_vec_normalize.reset()

    done = False

    while not done:
        action, _states = online_model.predict(obs, deterministic=deterministic)
        obs, _, done, _ = v_vec_normalize.step(action)
        suggested_weights = v_vec_normalize.get_attr('last_suggested_weights')[0]
        suggested_positions = v_vec_normalize.get_attr('last_suggested_positions')[0]

    return suggested_weights, suggested_positions


def run_model_qc(models_dir,
                 model_name,
                 online_algorithm,
                 deterministic,
                 data,
                 instruments,
                 env_lookback_period,
                 random_episode_start,
                 cash,
                 max_slippage_percent,
                 lot_size,
                 leverage,
                 pip_size,
                 pip_spread,
                 compute_position,
                 compute_indicators,
                 compute_reward,  # returns log_returns
                 meta_rl,
                 env_verbose):
    v_env = make_env_pf_fx(data,
                           instruments,
                           env_lookback_period,
                           random_episode_start,
                           cash,
                           max_slippage_percent,
                           lot_size,
                           leverage,
                           pip_size,
                           pip_spread,
                           compute_position,
                           compute_indicators,
                           compute_reward,  # returns log_returns
                           meta_rl,
                           env_verbose)

    #   check_env(v_env)

    v_online_model_dir = path_join(*[models_dir, model_name, 'online'])

    v_online_model_file_name = path_join(v_online_model_dir, 'model')
    v_online_model_file_name_stats = path_join(v_online_model_dir, 'stats.pkl')

    v_dummy_vec_env = DummyVecEnv([lambda: v_env])
    v_vec_normalize = VecNormalize.load(v_online_model_file_name_stats, v_dummy_vec_env)
    v_vec_normalize.training = False
    v_vec_normalize.norm_reward = False

    online_class = get_online_class_sb3(online_algorithm)
    online_model = online_class.load(v_online_model_file_name)

    online_model_parama = online_model.get_parameters()

    obs = v_vec_normalize.reset()

    done = False

    while not done:
        action, _states = online_model.predict(obs, deterministic=deterministic)
        obs, _, done, _ = v_vec_normalize.step(action)
        suggested_weights = v_vec_normalize.get_attr('last_suggested_weights')[0]
        suggested_positions = v_vec_normalize.get_attr('last_suggested_positions')[0]

    return suggested_weights, suggested_positions


def run_qc(cmd):
    process = psutil.Popen(cmd, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

    while True:

        output = process.stdout.readline()

        if 'Press any key to continue' in str(output.strip()):
            process.kill()
            break


# def run_qc_alert_on_error(cmd):
#     process = psutil.Popen(cmd, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
#
#     while True:
#
#         output = process.stdout.readline()
#
#         if 'error' in str(output.strip()).lower():
#             send_whatsup_message('Error in ')


def run_qc_get_stats(cmd):
    # process = psutil.Popen(cmd, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

    if platform.system() == 'Windows':
        process = psutil.Popen([cmd], stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    if platform.system() == 'Linux':
        process = psutil.Popen(['nohup ' + cmd], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setpgrp)

    stats = {}
    equity = {}

    while True:

        output = process.stdout.readline()

        if 'STATISTICS::' in str(output.strip()):
            split = output.split()
            split = [x.decode('utf-8') for x in split]
            key = ','.join(split[1:-1]).replace(',', ' ')
            value = split[-1]
            stats[key] = value

        if 'Press any key to continue' in str(output.strip()):
            #   print(f'pid = {process.pid}')
            process.kill()
            break

    return stats, equity


def run_qc_get_stats_with_params(test_script_dir, test_script, test_dir, test_exe, debug, **kwargs):
    # process = psutil.Popen(cmd, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

    # print(test_script_dir)
    # print(test_script)
    # print(test_dir)
    # print(test_exe)

    os.chdir(test_script_dir)

    for key, value in kwargs.items():
        #   print("%s == %s" % (key, value))
        #   print(f'Key = {key}, Value = {value}')
        replace_in_file(test_script, key + ' ', value)

    os.chdir(test_dir)

    if platform.system() == 'Windows':
        process = psutil.Popen([test_exe], stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    if platform.system() == 'Linux':
        process = psutil.Popen(['dotnet ' + test_exe], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setpgrp)

    stats = {}

    while True:

        output = process.stdout.readline()

        if debug:
            print(f'Reading line {i}, Output is {output}')

        if 'STATISTICS::' in str(output.strip()):
            split = output.split()
            split = [x.decode('utf-8') for x in split]
            key = ','.join(split[1:-1]).replace(',', ' ')
            value = split[-1]
            stats[key] = value

        if 'Press any key to continue' in str(output.strip()):
            #   print(f'pid = {process.pid}')
            process.kill()
            break

    #   print(f'STATS = {stats}')

    return stats
