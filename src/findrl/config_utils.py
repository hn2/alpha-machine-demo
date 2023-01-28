# https://docs.python.org/3/library/configparser.html

import ast
import configparser
from os.path import join as path_join

from .general_utils import convert_list_to_dict


def _read_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    return config


def get_paths_windows_params(config_file):
    config = _read_config(config_file)

    files_dir = config.get('paths_windows', 'files_dir')
    main_dir = config.get('paths_windows', 'main_dir')
    main_code_dir = path_join(main_dir, config.get('paths_windows', 'main_code_dir'))
    main_code_dir_train = path_join(main_code_dir, config.get('paths_windows', 'main_code_dir_train'))
    main_code_dir_test = path_join(main_code_dir, config.get('paths_windows', 'main_code_dir_test'))
    main_code_dir_tune = path_join(main_code_dir, config.get('paths_windows', 'main_code_dir_tune'))
    models_dir = path_join(files_dir, config.get('paths_windows', 'models_dir'))
    logs_dir = path_join(files_dir, config.get('paths_windows', 'logs_dir'))
    data_dir = path_join(main_dir, config.get('paths_windows', 'data_dir'))
    stats_dir = path_join(files_dir, config.get('paths_windows', 'stats_dir'))

    return files_dir, main_dir, main_code_dir, main_code_dir_train, main_code_dir_test, main_code_dir_tune, models_dir, logs_dir, data_dir, stats_dir


def get_paths_linux_params(config_file):
    config = _read_config(config_file)

    files_dir = config.get('paths_linux', 'files_dir')
    main_dir = config.get('paths_linux', 'main_dir')
    main_code_dir = path_join(main_dir, config.get('paths_linux', 'main_code_dir'))
    main_code_dir_train = path_join(main_code_dir, config.get('paths_linux', 'main_code_dir_train'))
    main_code_dir_test = path_join(main_code_dir, config.get('paths_linux', 'main_code_dir_test'))
    main_code_dir_tune = path_join(main_code_dir, config.get('paths_linux', 'main_code_dir_tune'))
    models_dir = path_join(files_dir, config.get('paths_linux', 'models_dir'))
    logs_dir = path_join(files_dir, config.get('paths_linux', 'logs_dir'))
    data_dir = path_join(main_dir, config.get('paths_linux', 'data_dir'))
    stats_dir = path_join(files_dir, config.get('paths_linux', 'stats_dir'))

    return files_dir, main_dir, main_code_dir, main_code_dir_train, main_code_dir_test, main_code_dir_tune, models_dir, logs_dir, data_dir, stats_dir


def get_train_params(config_file):
    config = _read_config(config_file)

    number_of_trials = config.getint('train', 'number_of_trials')
    #   train_look_back_period = config.getint('train', 'train_look_back_period')
    train_look_back_period = ast.literal_eval(config.get('train', 'train_look_back_period'))
    offset = config.getint('train', 'offset')
    total_timesteps = config.getint('train', 'total_timesteps')
    log_interval = config.getint('train', 'log_interval')
    subdir = config.get('train', 'subdir')
    market = config.get('train', 'market')
    resolution = config.get('train', 'resolution')
    instruments_in_portfolio = ast.literal_eval(config.get('train', 'instruments_in_portfolio'))
    num_of_instruments = ast.literal_eval(config.get('train', 'num_of_instruments'))
    num_of_instruments_in_portfolio = ast.literal_eval(config.get('train', 'num_of_instruments_in_portfolio'))
    is_equal = config.getboolean('train', 'is_equal')
    spread = config.getint('train', 'spread')

    return number_of_trials, subdir, train_look_back_period, offset, total_timesteps, log_interval, market, resolution, instruments_in_portfolio, num_of_instruments, num_of_instruments_in_portfolio, is_equal, spread


def get_test_params(config_file):
    config = _read_config(config_file)

    resolution = config.get('test', 'resolution')
    subdir = config.get('test', 'subdir')
    include_patterns = ast.literal_eval(config.get('test', 'include_patterns'))
    exclude_patterns = ast.literal_eval(config.get('test', 'exclude_patterns'))
    market_name = config.get('test', 'market_name')
    brokerage_name = config.get('test', 'brokerage_name')
    stat_file = config.get('test', 'stat_file')
    stat_file_head = config.getint('test', 'stat_file_head')
    files_lookback_hours = config.getint('test', 'files_lookback_hours')
    num_of_lookback_bars = config.getint('test', 'num_of_lookback_bars')
    num_of_lookback_bars_offset = config.getint('test', 'num_of_lookback_bars_offset')
    sort_column = config.get('test', 'sort_column')

    return resolution, subdir, include_patterns, exclude_patterns, market_name, brokerage_name, stat_file, stat_file_head, files_lookback_hours, num_of_lookback_bars, num_of_lookback_bars_offset, sort_column


def get_trade_params(config_file):
    config = _read_config(config_file)

    algorithms = ast.literal_eval(config.get('trade', 'algorithms'))
    sort_columns = config.get('trade', 'sort_columns')
    stat_column = config.get('trade', 'stat_column')
    position = config.getint('trade', 'position')
    number = config.getint('trade', 'number')
    trade_oanda = config.getboolean('trade', 'trade_oanda')
    rebalance_oanda = config.getboolean('trade', 'rebalance_oanda')
    liquidate_oanda = config.getboolean('trade', 'liquidate_oanda')
    trade_fxcm = config.getboolean('trade', 'trade_fxcm')
    delete_pending_orders_fxcm = config.getboolean('trade', 'delete_pending_orders_fxcm')
    rebalance_fxcm = config.getboolean('trade', 'rebalance_fxcm')
    liquidate_fxcm = config.getboolean('trade', 'liquidate_fxcm')
    upload_dropbox = config.getboolean('trade', 'upload_dropbox')
    dropbox_file_name = config.get('trade', 'dropbox_file_name')
    dropbox_remote_dir = config.get('trade', 'dropbox_remote_dir')

    return algorithms, sort_columns, stat_column, position, number, trade_oanda, rebalance_oanda, liquidate_oanda, trade_fxcm, delete_pending_orders_fxcm, rebalance_fxcm, liquidate_fxcm, upload_dropbox, dropbox_file_name, dropbox_remote_dir


def get_pf_fx_env_params_train(config_file):
    config = _read_config(config_file)

    env_lookback_periods = ast.literal_eval(config.get('pf_fx_env', 'env_lookback_period'))
    random_episode_start = config.getboolean('pf_fx_env', 'random_episode_start')
    cash = config.getfloat('pf_fx_env', 'cash')
    max_slippage_percent = config.getfloat('pf_fx_env', 'max_slippage_percent')
    lot_size = config.get('pf_fx_env', 'lot_size')
    leverage = config.getint('pf_fx_env', 'leverage')
    #   compute_position = ast.literal_eval(config.get('pf_fx_env', 'compute_position'))
    compute_position = config.get('pf_fx_env', 'compute_position')
    compute_indicators = ast.literal_eval(config.get('pf_fx_env', 'compute_indicators'))
    compute_reward = ast.literal_eval(config.get('pf_fx_env', 'compute_reward'))
    #   compute_reward = config.get('pf_fx_env', 'compute_reward')
    meta_rl = ast.literal_eval(config.get('pf_fx_env', 'meta_rl'))
    env_verbose = config.getboolean('pf_fx_env', 'env_verbose')
    num_of_envs = config.getint('pf_fx_env', 'num_of_envs')

    return env_lookback_periods, random_episode_start, cash, max_slippage_percent, lot_size, leverage, compute_position, compute_indicators, compute_reward, meta_rl, env_verbose, num_of_envs


def get_pf_fx_env_qc_params_train(config_file):
    config = _read_config(config_file)

    random_episode_start = config.getboolean('pf_fx_env_qc', 'random_episode_start')
    env_lookback_periods = ast.literal_eval(config.get('pf_fx_env_qc', 'env_lookback_period'))
    #   compute_position = ast.literal_eval(config.get('pf_fx_env', 'compute_position'))
    compute_position = config.get('pf_fx_env_qc', 'compute_position')
    compute_indicators = ast.literal_eval(config.get('pf_fx_env_qc', 'compute_indicators'))
    compute_reward = ast.literal_eval(config.get('pf_fx_env_qc', 'compute_reward'))
    #   compute_reward = config.get('pf_fx_env', 'compute_reward')
    meta_rl = ast.literal_eval(config.get('pf_fx_env', 'meta_rl'))
    env_verbose = config.getboolean('pf_fx_env_qc', 'env_verbose')
    num_of_envs = config.getint('pf_fx_env_qc', 'num_of_envs')

    return random_episode_start, env_lookback_periods, compute_position, compute_indicators, compute_reward, meta_rl, env_verbose, num_of_envs


def get_agent_params_sb3(config_file):
    config = _read_config(config_file)

    algorithms = ast.literal_eval(config.get('agent', 'algorithms'))
    model_verbose = config.getboolean('agent', 'model_verbose')
    callback_verbose = config.getboolean('agent', 'callback_verbose')
    save_replay_buffer = config.getboolean('agent', 'save_replay_buffer')
    use_tensorboard = config.getboolean('agent', 'use_tensorboard')
    use_callback = config.getboolean('agent', 'use_callback')
    check_freq = config.getint('agent', 'check_freq')
    callback_lookback = config.getint('agent', 'callback_lookback')
    save_freq = config.getint('agent', 'save_freq')
    params = convert_list_to_dict(ast.literal_eval(config.get('agent', 'params')))
    learning_rate = ast.literal_eval(config.get('agent', 'learning_rate'))
    batch_sizes = ast.literal_eval(config.get('agent', 'batch_size'))
    net_arch = ast.literal_eval(config.get('agent', 'net_arch'))
    use_linear_schedule = config.getboolean('agent', 'use_linear_schedule')
    action_noise = ast.literal_eval(config.get('agent', 'action_noise'))
    #   none   normalactionnoise   ornsteinuhlenbeckactionnoise
    noise_sigma = config.getfloat('agent', 'noise_sigma')
    use_sde = ast.literal_eval(config.get('agent', 'use_sde'))

    return algorithms, model_verbose, callback_verbose, save_replay_buffer, use_tensorboard, use_callback, check_freq, \
           callback_lookback, save_freq, params, learning_rate, batch_sizes, net_arch, use_linear_schedule, action_noise, noise_sigma, use_sde


def get_agent_params_d3rlpy(config_file):
    config = _read_config(config_file)

    algorithms = ast.literal_eval(config.get('agent', 'algorithms'))
    model_verbose = config.getboolean('agent', 'model_verbose')
    save_replay_buffer = config.getboolean('agent', 'save_replay_buffer')
    use_tensorboard = config.getboolean('agent', 'use_tensorboard')
    learning_rate = ast.literal_eval(config.get('agent', 'learning_rate'))
    batch_sizes = ast.literal_eval(config.get('agent', 'batch_size'))
    hidden_units = ast.literal_eval(config.get('agent', 'hidden_units'))
    activation = ast.literal_eval(config.get('agent', 'activation'))
    use_batch_norm = ast.literal_eval(config.get('agent', 'use_batch_norm'))
    use_dense = ast.literal_eval(config.get('agent', 'use_dense'))
    preprocessor = config.get('agent', 'preprocessor')

    return algorithms, model_verbose, save_replay_buffer, use_tensorboard, learning_rate, batch_sizes, hidden_units, activation, use_batch_norm, use_dense, preprocessor
    # return 1 ,2 ,3 ,4, 5


def get_choose_best_model_agent_params(config_file):
    config = _read_config(config_file)

    model_verbose = config.getboolean('choose_best_model_agent', 'model_verbose')
    callback_verbose = config.getboolean('choose_best_model_agent', 'callback_verbose')
    save_replay_buffer = config.getboolean('choose_best_model_agent', 'save_replay_buffer')
    use_tensorboard = config.getboolean('choose_best_model_agent', 'use_tensorboard')
    use_callback = config.getboolean('choose_best_model_agent', 'use_callback')
    check_freq = config.getint('choose_best_model_agent', 'check_freq')
    callback_lookback = config.getint('choose_best_model_agent', 'callback_lookback')
    save_freq = config.getint('choose_best_model_agent', 'save_freq')
    params = convert_list_to_dict(ast.literal_eval(config.get('choose_best_model_agent', 'params')))
    batch_size = ast.literal_eval(config.get('agent', 'algorithms'))
    net_arch = ast.literal_eval(config.get('choose_best_model_agent', 'net_arch'))
    use_linear_schedule = config.getboolean('choose_best_model_agent', 'use_linear_schedule')
    action_noise = ast.literal_eval(config.get('choose_best_model_agent', 'action_noise'))
    #   none   normalactionnoise   ornsteinuhlenbeckactionnoise
    noise_sigma = config.getfloat('choose_best_model_agent', 'noise_sigma')
    use_sde = config.getboolean('choose_best_model_agent', 'use_sde')

    return model_verbose, callback_verbose, save_replay_buffer, use_tensorboard, use_callback, check_freq, \
           callback_lookback, save_freq, params, batch_size, net_arch, use_linear_schedule, action_noise, noise_sigma, use_sde


def get_model_params(config_file):
    config = _read_config(config_file)

    delimeter = config.get('model', 'delimeter')
    model_prefix = config.get('model', 'model_prefix')

    return delimeter, model_prefix


def get_offline_train_params(config_file):
    config = _read_config(config_file)

    number_of_trials = config.getint('offline_train', 'number_of_trials')
    #   train_look_back_period = config.getint('offline_train', 'train_look_back_period')
    train_look_back_period = ast.literal_eval(config.get('offline_train', 'train_look_back_period'))
    total_timesteps = config.getint('offline_train', 'total_timesteps')
    log_interval = config.getint('offline_train', 'log_interval')
    subdir = config.get('offline_train', 'subdir')
    market = config.get('offline_train', 'market')
    resolution = config.get('offline_train', 'resolution')
    num_of_instruments = ast.literal_eval(config.get('offline_train', 'num_of_instruments'))
    num_of_instruments_in_portfolio = ast.literal_eval(config.get('offline_train', 'num_of_instruments_in_portfolio'))
    spread = config.getint('offline_train', 'spread')

    return number_of_trials, subdir, train_look_back_period, total_timesteps, log_interval, market, resolution, num_of_instruments, num_of_instruments_in_portfolio, spread


def get_optuna_params(config_file):
    config = _read_config(config_file)

    num_of_instruments = ast.literal_eval(config.get('optuna', 'num_of_instruments'))
    spread = config.getint('optuna', 'spread')
    market = config.get('optuna', 'market')
    subdir = config.get('optuna', 'subdir')
    resolution = config.get('optuna', 'resolution')
    #   train_look_back_period = config.getint('optuna', 'train_look_back_period')
    train_look_back_period = ast.literal_eval(config.get('optuna', 'train_look_back_period'))
    train_test_split = config.getfloat('optuna', 'train_test_split')
    connection_url = config.get('optuna', 'connection_url')

    return num_of_instruments, spread, market, subdir, resolution, train_look_back_period, train_test_split, connection_url


def get_train_choose_best_model_params(config_file):
    config = _read_config(config_file)

    algorithms = ast.literal_eval(config.get('train_choose_best_model', 'algorithms'))
    number_of_trials = config.getint('train_choose_best_model', 'number_of_trials')
    #   train_look_back_period = config.getint('train_choose_best_model', 'train_look_back_period')
    train_look_back_period = ast.literal_eval(config.get('train_choose_best_model', 'train_look_back_period'))
    total_timesteps = config.getint('train_choose_best_model', 'total_timesteps')
    subdir = config.get('train_choose_best_model', 'subdir')
    subdir_choose_best_model = config.get('train_choose_best_model', 'subdir_choose_best_model')
    train_test_split = config.getfloat('train_choose_best_model', 'train_test_split')
    market = config.get('train_choose_best_model', 'market')
    resolution = config.get('train_choose_best_model', 'resolution')
    num_of_instruments = ast.literal_eval(config.get('train_choose_best_model', 'num_of_instruments'))
    spread = config.getint('train_choose_best_model', 'spread')
    files_lookback_hours = config.getint('train_choose_best_model', 'files_lookback_hours')
    include_patterns = ast.literal_eval(config.get('train_choose_best_model', 'include_patterns'))
    exclude_patterns = ast.literal_eval(config.get('train_choose_best_model', 'exclude_patterns'))
    delimeter = config.get('train_choose_best_model', 'delimeter')
    deterministic = config.getboolean('train_choose_best_model', 'deterministic')

    return algorithms, number_of_trials, subdir, subdir_choose_best_model, train_test_split, train_look_back_period, total_timesteps, market, resolution, num_of_instruments, spread, files_lookback_hours, include_patterns, exclude_patterns, delimeter, deterministic


def get_test_ensamble_params(config_file):
    config = _read_config(config_file)

    models_dir = config.get('test-ensamble', 'models_dir')
    resolution = config.get('test-ensamble', 'resolution')
    subdir = config.get('test-ensamble', 'subdir')
    test_script = config.get('test', 'test_script')
    test_exe = config.get('test', 'test_exe')
    include_patterns = ast.literal_eval(config.get('test-ensamble', 'include_patterns'))
    exclude_patterns = ast.literal_eval(config.get('test-ensamble', 'exclude_patterns'))
    market_name = config.get('test-ensamble', 'market_name')
    brokerage_name = config.get('test-ensamble', 'brokerage_name')
    stat_file = config.get('test', 'stat_file')
    stat_file_head = config.getint('test', 'stat_file_head')
    files_lookback_hours = config.getint('test', 'files_lookback_hours')
    num_of_lookback_bars = config.getint('test', 'num_of_lookback_bars')
    num_of_lookback_bars_offset = config.getint('test', 'num_of_lookback_bars_offset')
    sort_column = config.get('test', 'sort_column')

    return resolution, subdir, test_script, test_exe, include_patterns, exclude_patterns, market_name, brokerage_name, stat_file, stat_file_head, files_lookback_hours, num_of_lookback_bars, num_of_lookback_bars_offset, sort_column


# def get_trade_params(config_file):
#     config = _read_config(config_file)
#
#     algorithms = ast.literal_eval(config.get('trade', 'algorithms'))
#     sort_columns = config.get('trade', 'sort_columns')
#     stat_column = config.get('trade', 'stat_column')
#     head = config.getint('trade', 'head')
#     #   number = config.getint('trade', 'number')
#     account_currency = config.get('trade', 'account_currency')
#     upload_dropbox = config.getboolean('trade', 'upload_dropbox')
#     trade_oanda = config.getboolean('trade', 'trade_fxcm')
#     rebalance_oanda = config.getboolean('trade', 'rebalance_fxcm')
#     trade_fxcm = config.getboolean('trade', 'trade_fxcm')
#     rebalance_fxcm = config.getboolean('trade', 'rebalance_fxcm')
#     dropbox_local_dir = config.get('trade', 'dropbox_local_dir')
#     dropbox_file_name = config.get('trade', 'dropbox_file_name')
#     dropbox_remote_dir = config.get('trade', 'dropbox_remote_dir')
#
#     return algorithms, sort_columns, stat_column, head, account_currency, upload_dropbox, trade_oanda, rebalance_oanda, trade_fxcm, rebalance_fxcm, dropbox_local_dir, dropbox_file_name, dropbox_remote_dir


def get_download_params(config_file):
    config = _read_config(config_file)

    lookback = config.getint('download', 'lookback')
    num_of_instruments = config.get('download', 'num_of_instruments')
    markets = ast.literal_eval(config.get('download', 'markets'))
    resolutions = ast.literal_eval(config.get('download', 'resolutions'))
    dest_dir_format = config.get('download', 'dest_dir_format')
    seconds_to_sleep = config.get('download', 'seconds_to_sleep')

    return lookback, num_of_instruments, markets, resolutions, dest_dir_format, seconds_to_sleep


def get_tokens_params(config_file):
    config = _read_config(config_file)

    fxcm_access_token_real_1 = config.get('tokens', 'fxcm_access_token_real_1')
    fxcm_access_token_real_2 = config.get('tokens', 'fxcm_access_token_real_2')
    fxcm_access_token_demo_1 = config.get('tokens', 'fxcm_access_token_demo_1')
    fxcm_access_token_demo_2 = config.get('tokens', 'fxcm_access_token_demo_2')
    fxcm_access_token_demo_3 = config.get('tokens', 'fxcm_access_token_demo_3')
    fxcm_access_token_demo_4 = config.get('tokens', 'fxcm_access_token_demo_4')
    fxcm_access_token_demo_5 = config.get('tokens', 'fxcm_access_token_demo_5')
    oanda_access_token = config.get('tokens', 'oanda_access_token')
    dropbox_access_token = config.get('tokens', 'dropbox_access_token')
    github_access_token = config.get('tokens', 'github_access_token')
    aws_server_public_key = config.get('tokens', 'aws_server_public_key')
    aws_server_secret_key = config.get('tokens', 'aws_server_secret_key')

    return fxcm_access_token_real_1, fxcm_access_token_real_2, fxcm_access_token_demo_1, fxcm_access_token_demo_2, fxcm_access_token_demo_3, fxcm_access_token_demo_4, fxcm_access_token_demo_5, oanda_access_token, dropbox_access_token, github_access_token, aws_server_public_key, aws_server_secret_key
