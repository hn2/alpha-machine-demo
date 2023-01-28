import platform
from datetime import datetime, timedelta
from os.path import join as path_join

import pandas as pd

from findrl.config_utils import get_test_params
from findrl.file_utils import get_subfolders, get_timestamps, get_instruments_in_model
from findrl.model_utils import get_patterns_list, parse_model_name_online_algorithm, \
    parse_model_name_train_lookback_period
from findrl.run_utils import run_qc_get_stats_with_params


def main():
    v_resolution, v_subdir, v_include_patterns, v_exclude_patterns, v_market_name, v_brokerage_name, v_stat_file, v_stat_file_head, v_files_lookback_hours, v_num_of_lookback_bars, v_num_of_lookback_bars_offset_ini_file, v_sort_column = get_test_params(
        CONFIG_FILE)

    print(f'Test script dir: {TEST_SCRIPT_DIR}')
    print(f'Test script: {TEST_SCRIPT}')
    print(f'Test dir: {TEST_DIR}')
    print(f'Test exe: {TEST_EXE}')
    print(f'Models Dir: {MODELS_DIR}')
    print(f'Lookback Bars: {v_num_of_lookback_bars}')
    print(f'Resolution: {v_resolution}')

    if v_stat_file:
        v_include_patterns = get_patterns_list(v_stat_file, v_stat_file_head)

    v_subfolders = get_subfolders(MODELS_DIR, v_files_lookback_hours, v_include_patterns)

    v_timestamps = get_timestamps(MODELS_DIR, v_files_lookback_hours, v_include_patterns)

    if platform.system() == 'Windows':
        v_models = [x.split('\\')[-1] for x in v_subfolders]
    elif platform.system() == 'Linux':
        v_models = [x.split('/')[-1] for x in v_subfolders]

    df_stats = pd.DataFrame(
        columns=['Timestamp', 'Total Trades', 'Average Win', 'Average Loss', 'Compounding Annual Return',
                 'Drawdown', 'Expectancy', 'Net Profit', 'Sharpe Ratio',
                 'Probabilistic Sharpe Ratio', 'Loss Rate', 'Win Rate', 'Profit-Loss Ratio',
                 'Alpha', 'Beta', 'Annual Standard Deviation', 'Annual Variance',
                 'Information Ratio', 'Tracking Error', 'Treynor Ratio', 'Total Fees',
                 'Fitness Score', 'Kelly Criterion Estimate', 'Kelly Criterion Probability Value',
                 'Sortino Ratio', 'Return Over Maximum Drawdown', 'Portfolio Turnover',
                 'Total Insights Generated',
                 'Total Insights Closed', 'Total Insights Analysis Completed',
                 'Long Insight Count', 'Short Insight Count',
                 'Long/Short Ratio', 'Estimated Monthly Alpha Value',
                 'Total Accumulated Estimated Alpha Value',
                 'Mean Population Estimated Insight Value', 'Mean Population Direction',
                 'Mean Population Magnitude', 'Rolling Averaged Population Direction',
                 'Rolling Averaged Population Magnitude', 'OrderListHash', 'Instruments'])

    num_of_models = len(v_models)

    for i, (model, timestamp) in enumerate(zip(v_models, v_timestamps)):
        #   for model, timestamp in zip(v_models, v_timestamps):
        print(f'Using model number {i}/{num_of_models}: {model}')
        print(f'Using timestamp: {timestamp}')

        try:
            v_delimeter = "-"
            v_train_lookback_period = parse_model_name_train_lookback_period(model, v_delimeter)

            if v_num_of_lookback_bars_offset_ini_file == -1:
                v_num_of_lookback_bars_offset = int(v_train_lookback_period)
            else:
                v_num_of_lookback_bars_offset = v_num_of_lookback_bars_offset_ini_file

            print(f'Lookback Bars Offset: {v_num_of_lookback_bars_offset}')

            if v_resolution == 'daily' and v_num_of_lookback_bars > 0:
                end = datetime.now() - timedelta(days=v_num_of_lookback_bars_offset)
                start = end - timedelta(days=v_num_of_lookback_bars)
            elif v_resolution == 'hour' and v_num_of_lookback_bars > 0:
                end = datetime.now() - timedelta(days=round(v_num_of_lookback_bars_offset / 24))
                start = end - timedelta(days=round(v_num_of_lookback_bars / 24))

            v_start_date = [start.year, start.month, start.day]
            v_end_date = [end.year, end.month, end.day]
            v_lot_size = "Micro"
            v_leverage = 20
            v_max_slippage_percent = 1e-2
            v_cash = 1e3

            dict_stats = run_qc_get_stats_with_params(test_script_dir=TEST_SCRIPT_DIR,
                                                      test_script=TEST_SCRIPT,
                                                      test_dir=TEST_DIR,
                                                      test_exe=TEST_EXE,
                                                      debug=False,
                                                      MODELS_DIR=f'MODELS_DIR = "{MODELS_DIR}"\n',
                                                      MODEL_NAME=f'MODEL_NAME = "{model}"\n',
                                                      DELIMITER=f'DELIMITER = "{v_delimeter}"\n',
                                                      START_DATE=f'START_DATE = {v_start_date}\n',
                                                      END_DATE=f'END_DATE = {v_end_date}\n',
                                                      LOT_SIZE=f'LOT_SIZE = "{v_lot_size}"\n',
                                                      LEVERAGE=f'LEVERAGE = {v_leverage}\n',
                                                      MAX_SLIPPAGE_PERCENT=f'MAX_SLIPPAGE_PERCENT = {v_max_slippage_percent}\n',
                                                      CASH=f'CASH = {v_cash}\n')

            row_stats = pd.Series(dict_stats, name=model)
            row_stats['Timestamp'] = timestamp

            try:
                row_stats['Instruments'] = get_instruments_in_model(MODELS_DIR, model)
            except Exception as e:
                row_stats['Instruments'] = str(e)

            row_stats['Model Name'] = chr(39) + model + chr(39) + chr(44)
            row_stats['algo'] = parse_model_name_online_algorithm(model, v_delimeter)

            df_stats = df_stats.append(row_stats)


        except Exception as e:

            print(e)

    now = datetime.now()
    d, m, y = now.day, now.month, now.year

    df_stats['sort_column'] = df_stats[v_sort_column].str.strip('%').astype(float)
    df_stats_sorted = df_stats.sort_values(['algo', 'sort_column'], ascending=False)

    #   Save stats to xlsx, csv
    if v_resolution == 'daily':
        stats_file_name = path_join(STATS_DIR,
                                    #   f'statistics-{MARKET_NAME}-{BROKERAGE_NAME}-{d}-{m}-{y}-{NUM_OF_DAYS}-{INCLUDE_PATTERNS.replace("","all")}.xlsx')
                                    f'stats_1_lev_{v_leverage}-{v_subdir}-{v_market_name}-{v_brokerage_name}-{v_resolution}-{d}-{m}-{y}-{v_num_of_lookback_bars_offset}-{v_num_of_lookback_bars}-{v_include_patterns[0]}.xlsx')

        df_stats_sorted.to_excel(stats_file_name, index=True, header=True)

        print(f'Stats save to {stats_file_name}')

        stats_file_name = path_join(STATS_DIR,
                                    #   f'statistics-{MARKET_NAME}-{BROKERAGE_NAME}-{d}-{m}-{y}-{NUM_OF_DAYS}-{INCLUDE_PATTERNS.replace("","all")}.xlsx')
                                    f'stats_1_lev_{v_leverage}-{v_subdir}-{v_market_name}-{v_brokerage_name}-{v_resolution}-{d}-{m}-{y}-{v_num_of_lookback_bars_offset}-{v_num_of_lookback_bars}-{v_include_patterns[0]}.csv')

        df_stats_sorted.to_csv(stats_file_name, index=True, header=True)

        print(f'Stats save to {stats_file_name}')

    elif v_resolution == 'hour':
        stats_file_name = path_join(STATS_DIR,
                                    #   f'statistics-{MARKET_NAME}-{BROKERAGE_NAME}-{d}-{m}-{y}-{NUM_OF_DAYS}-{INCLUDE_PATTERNS.replace("","all")}.xlsx')
                                    f'stats_1_lev_{v_leverage}-{v_subdir}-{v_market_name}-{v_brokerage_name}-{v_resolution}-{d}-{m}-{y}-{v_num_of_lookback_bars_offset}-{v_num_of_lookback_bars}-{v_include_patterns[0]}.xlsx')

        df_stats_sorted.to_excel(stats_file_name, index=True, header=True)

        print(f'Stats save to {stats_file_name}')

        stats_file_name = path_join(STATS_DIR,
                                    #   f'statistics-{MARKET_NAME}-{BROKERAGE_NAME}-{d}-{m}-{y}-{NUM_OF_DAYS}-{INCLUDE_PATTERNS.replace("","all")}.xlsx')
                                    f'stats_1_lev_{v_leverage}-{v_subdir}-{v_market_name}-{v_brokerage_name}-{v_resolution}-{d}-{m}-{y}-{v_num_of_lookback_bars_offset}-{v_num_of_lookback_bars}-{v_include_patterns[0]}.csv')

        df_stats_sorted.to_csv(stats_file_name, index=True, header=True)

        print(f'Stats save to {stats_file_name}')


if __name__ == "__main__":
    main()
