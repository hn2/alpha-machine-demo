import os
import platform
from datetime import datetime, timedelta
from os.path import join as path_join

import pandas as pd

from findrl.config_utils import get_paths_windows_params, get_paths_linux_params, get_test_params
from findrl.file_utils import get_subfolders, get_timestamps, replace_in_file
from findrl.model_utils import parse_model_name_online_algorithm, get_patterns_list
from findrl.run_utils import run_qc_get_stats

DELIMITER = '-'
LEVERAGE = 20


def main():
    config_file = '../settings/config-test-oanda-daily-online-batch.ini'
    if platform.system() == 'Windows':
        v_main_dir, v_main_code_dir, v_main_code_dir_train, v_main_code_dir_test, v_main_code_dir_tune, v_models_dir, v_logs_dir, v_data_dir, v_test_dir, v_stats_dir = get_paths_windows_params(
            config_file)
    if platform.system() == 'Linux':
        v_main_dir, v_main_code_dir, v_main_code_dir_train, v_main_code_dir_test, v_main_code_dir_tune, v_models_dir, v_logs_dir, v_data_dir, v_test_dir, v_stats_dir = get_paths_linux_params(
            config_file)
    v_resolution, v_subdir, v_test_script, v_test_exe, v_include_patterns, v_exclude_patterns, v_market_name, v_brokerage_name, v_stat_file, v_stat_file_head, v_end_day_offset, v_files_lookback_hours, v_num_of_lookback_days_daily, v_num_of_lookback_days_hour, v_sort_column = get_test_params(
        config_file)

    v_test_script = path_join(v_main_code_dir_test, v_test_script)
    v_models_dir = path_join(*[v_models_dir, v_resolution, v_subdir])

    print(f'v_test_script: {v_test_script}')
    print(f'MODELS_DIR: {v_models_dir}')

    print(v_resolution)

    if v_stat_file:
        v_include_patterns = get_patterns_list(v_stat_file, v_stat_file_head)

    v_subfolders = get_subfolders(v_models_dir, v_files_lookback_hours, v_include_patterns)

    v_timestamps = get_timestamps(v_models_dir, v_files_lookback_hours, v_include_patterns)

    v_models = [x.split('\\')[-1] for x in v_subfolders]

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
                 'Rolling Averaged Population Magnitude', 'OrderListHash', 'Model Name'])

    print(f'v_test_dir: {v_test_dir}')
    os.chdir(v_test_dir)

    # v_brokerage_name = 'FxcmBrokerage'

    replace_in_file(v_test_script, "self.SetBrokerageModel",
                    f"        self.SetBrokerageModel(BrokerageName.{v_brokerage_name})\n")

    replace_in_file(v_test_script, "LEVERAGE",
                    f"LEVERAGE = {LEVERAGE}\n")

    if v_resolution == 'daily' and v_num_of_lookback_days_daily > 0:
        end = datetime.now() - timedelta(days=v_end_day_offset)
        start = end - timedelta(days=v_num_of_lookback_days_daily)

        # start date
        replace_in_file(v_test_script, "self.SetStartDate",
                        f"        self.SetStartDate({start.year}, {start.month}, {start.day})\n")

        # end date
        replace_in_file(v_test_script, "self.SetEndDate",
                        f"        self.SetEndDate({end.year}, {end.month}, {end.day})\n")

    elif v_resolution == 'hour' and v_num_of_lookback_days_hour > 0:
        end = datetime.now() - timedelta(days=v_end_day_offset)
        start = end - timedelta(days=v_num_of_lookback_days_hour)

        # start date
        replace_in_file(v_test_script, "self.SetStartDate",
                        f"        self.SetStartDate({start.year}, {start.month}, {start.day})\n")

        # end date
        replace_in_file(v_test_script, "self.SetEndDate",
                        f"        self.SetEndDate({end.year}, {end.month}, {end.day})\n")

    num_of_models = len(v_models)

    for i, (model, timestamp) in enumerate(zip(v_models, v_timestamps)):
        #   for model, timestamp in zip(v_models, v_timestamps):
        print(f'Using model number {i}/{num_of_models}: {model}')
        print(f'Using timestamp: {timestamp}')

        try:

            replace_in_file(v_test_script, "MODELS_DIR = ", f"MODELS_DIR = r'{v_models_dir}'\n")
            replace_in_file(v_test_script, "MODEL_NAME = ", f"MODEL_NAME = '{model}'\n")

            dict_stats, dict_equity = run_qc_get_stats(v_test_exe)

            row_stats = pd.Series(dict_stats, name=model)
            row_stats['Timestamp'] = timestamp
            row_stats['Model Name'] = chr(39) + model + chr(39) + chr(44)
            row_stats['algo'] = parse_model_name_online_algorithm(model, DELIMITER)

            df_stats = df_stats.append(row_stats)


        except Exception as e:

            print(e)

    now = datetime.now()
    d, m, y = now.day, now.month, now.year

    df_stats['sort_column'] = df_stats[v_sort_column].str.strip('%').astype(float)
    df_stats_sorted = df_stats.sort_values(['algo', 'sort_column'], ascending=False)

    #   Save stats to xlsx, csv
    if v_resolution == 'daily':
        stats_file_name = path_join(v_stats_dir,
                                    #   f'statistics-{MARKET_NAME}-{BROKERAGE_NAME}-{d}-{m}-{y}-{NUM_OF_DAYS}-{INCLUDE_PATTERNS.replace("","all")}.xlsx')
                                    f'statistics_leverage_{LEVERAGE}-{v_subdir}-{v_market_name}-{v_brokerage_name}-{v_resolution}-{d}-{m}-{y}-{v_end_day_offset}-{v_num_of_lookback_days_daily}-{v_include_patterns[0]}.xlsx')

        df_stats_sorted.to_excel(stats_file_name, index=True, header=True)

        print(f'Stats save to {stats_file_name}')

        stats_file_name = path_join(v_stats_dir,
                                    #   f'statistics-{MARKET_NAME}-{BROKERAGE_NAME}-{d}-{m}-{y}-{NUM_OF_DAYS}-{INCLUDE_PATTERNS.replace("","all")}.xlsx')
                                    f'statistics_leverage_{LEVERAGE}-{v_subdir}-{v_market_name}-{v_brokerage_name}-{v_resolution}-{d}-{m}-{y}-{v_end_day_offset}-{v_num_of_lookback_days_daily}-{v_include_patterns[0]}.csv')

        df_stats_sorted.to_csv(stats_file_name, index=True, header=True)

        print(f'Stats save to {stats_file_name}')

    elif v_resolution == 'hour':
        stats_file_name = path_join(v_stats_dir,
                                    #   f'statistics-{MARKET_NAME}-{BROKERAGE_NAME}-{d}-{m}-{y}-{NUM_OF_DAYS}-{INCLUDE_PATTERNS.replace("","all")}.xlsx')
                                    f'statistics_leverage_{LEVERAGE}-{v_subdir}-{v_market_name}-{v_brokerage_name}-{v_resolution}-{d}-{m}-{y}-{v_end_day_offset}-{v_num_of_lookback_days_hour}-{v_include_patterns[0]}.xlsx')

        df_stats_sorted.to_excel(stats_file_name, index=True, header=True)

        print(f'Stats save to {stats_file_name}')

        stats_file_name = path_join(v_stats_dir,
                                    #   f'statistics-{MARKET_NAME}-{BROKERAGE_NAME}-{d}-{m}-{y}-{NUM_OF_DAYS}-{INCLUDE_PATTERNS.replace("","all")}.xlsx')
                                    f'statistics_leverage_{LEVERAGE}-{v_subdir}-{v_market_name}-{v_brokerage_name}-{v_resolution}-{d}-{m}-{y}-{v_end_day_offset}-{v_num_of_lookback_days_hour}-{v_include_patterns[0]}.csv')

        df_stats_sorted.to_csv(stats_file_name, index=True, header=True)

        print(f'Stats save to {stats_file_name}')


if __name__ == "__main__":
    main()
