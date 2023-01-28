import os
from datetime import datetime, timedelta
from os.path import join as path_join

import pandas as pd

from findrl.config_utils import get_test_ensamble_params
from findrl.file_utils import get_dict_from_json, replace_in_file
from findrl.general_utils import get_name_from_dict_items
from findrl.run_utils import run_qc_get_stats

DELIMITER = '-'
LEVERAGE = 20
PARAM_FILE = r'C:\alpha-machine\src\drl-pf-fx-sb3\settings\test-oanda-ensamble-large.json'
STATS_FILE_NAME_PREFIX = 'statistics-ensamble_leverage'


def main():
    v_config_file = 'settings/config-oanda-train-test-daily.ini'
    v_models_dir, v_resolution, v_subdir, v_test_dir, v_test_script, v_test_exe, v_include_patterns, v_exclude_patterns, v_market_name, v_brokerage_name, v_stat_file, v_stat_file_head, v_end_day_offset, v_files_lookback_hours, v_num_of_lookback_days_daily, v_num_of_lookback_days_hour, v_stats_dir, v_sort_column = get_test_ensamble_params(
        v_config_file)

    df_stats = pd.DataFrame(
        columns=['Total Trades', 'Average Win', 'Average Loss', 'Compounding Annual Return',
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

    params = get_dict_from_json(PARAM_FILE)

    for key, value in params.items():
        print(f'Using param: {key}')
        print(f'Using value: {value}')

        try:

            replace_in_file(v_test_script, "self.params = ",
                            f"        self.params = {value}\n")

            dict_stats, dict_equity = run_qc_get_stats(v_test_exe)

            name = get_name_from_dict_items(value)

            row_stats = pd.Series(dict_stats, name=name)
            row_stats['Name'] = name

            df_stats = df_stats.append(row_stats)

        except Exception as e:

            print(e)

    now = datetime.now()
    d, m, y = now.day, now.month, now.year

    df_stats['sort_column'] = df_stats[v_sort_column].str.strip('%').astype(float)
    df_stats_sorted = df_stats.sort_values(['Name', 'sort_column'], ascending=False)

    print(df_stats_sorted)

    #   Save stats to xlsx, csv
    stats_file_name = path_join(v_stats_dir,
                                #   f'statistics-{MARKET_NAME}-{BROKERAGE_NAME}-{d}-{m}-{y}-{NUM_OF_DAYS}-{INCLUDE_PATTERNS.replace("","all")}.xlsx')
                                f'{STATS_FILE_NAME_PREFIX}-{LEVERAGE}-{v_subdir}-{v_market_name}-{v_brokerage_name}-{v_resolution}-{d}-{m}-{y}-{v_end_day_offset}-{v_num_of_lookback_days_daily}-{v_include_patterns[0]}.xlsx')

    df_stats_sorted.to_excel(stats_file_name, index=True, header=True)

    print(f'Stats save to {stats_file_name}')

    # stats_file_name = path_join(v_stats_dir,
    #                             #   f'statistics-{MARKET_NAME}-{BROKERAGE_NAME}-{d}-{m}-{y}-{NUM_OF_DAYS}-{INCLUDE_PATTERNS.replace("","all")}.xlsx')
    #                             f'{STATS_FILE_NAME_PREFIX}-{LEVERAGE}-{v_subdir}-{v_market_name}-{v_brokerage_name}-{v_resolution}-{d}-{m}-{y}-{v_end_day_offset}-{v_num_of_lookback_days_daily}-{v_include_patterns[0]}.csv')
    #
    # df_stats_sorted.to_csv(stats_file_name, index=True, header=True)

    print(f'Stats save to {stats_file_name}')


if __name__ == "__main__":
    main()
