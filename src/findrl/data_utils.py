#   https://github.com/alpha-machine/Lean/blob/master/data/forex/readme.md
#   https://realpython.com/python-f-strings/

from math import sqrt
from os.path import join as path_join

import fxcmpy
import matplotlib.pyplot as plt
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from .technical_features import technical_features


def time_format_changer_fxcm(date):
    return date.strftime('%Y%m%d %H:%M')


def time_format_changer_oanda(date):
    return date.replace('-', '').replace('.000000000Z', '').replace('T', ' ')[:-3]


def prepare_data_cluster(data_dir, market, resolution, pairs, period):
    #   data = np.empty(shape=(len(pairs), 0, 4), dtype=np.float64)

    column_names = ['datetime', 'BidOpen', 'BidHigh', 'BidLow', 'BidClose', 'LastBidSize', 'AskOpen', 'AskHigh',
                    'AskLow', 'AskClose', 'LastAskSize']

    all_history_dict = {}

    for i, pair in enumerate(pairs):
        file_name_zip = path_join(*[data_dir, market, resolution, pair + '.zip'])
        history = pd.read_csv(file_name_zip, compression='zip', delimiter=',',
                              parse_dates=True, names=column_names,
                              index_col='datetime',
                              header=None)

        history['Open'] = history[['BidOpen', 'AskOpen']].mean(axis=1)
        history['High'] = history[['BidHigh', 'AskHigh']].mean(axis=1)
        history['Low'] = history[['BidLow', 'AskLow']].mean(axis=1)
        history['Close'] = history[['BidClose', 'AskClose']].mean(axis=1)

        # all_history_dict[pair] = history[['Open', 'High', 'Low', 'Close']]
        all_history_dict[pair] = history[['Close']]

    #   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

    all_history_df = pd.concat([df for df in list(all_history_dict.values())], axis=1, join='inner')

    row_count = all_history_df.shape[0]
    start_row = row_count - period - 1
    start_date = all_history_df.index.values[start_row]
    end_row = row_count - 1
    end_date = all_history_df.index.values[end_row]

    print(
        f'row_count={row_count}, start_row={start_row}, start_date={start_date}, end_row={end_row}, end_date={end_date}')

    all_history_df.columns = pairs
    all_history_df.reset_index(inplace=True)
    all_history_df.rename(columns={'datetime': 'Date'}, inplace=True)

    return all_history_df


def prepare_data_train(data_dir, market, resolution, pairs, period):
    data = np.empty(shape=(len(pairs), 0, 4), dtype=np.float)

    column_names = ['datetime', 'BidOpen', 'BidHigh', 'BidLow', 'BidClose', 'LastBidSize', 'AskOpen', 'AskHigh',
                    'AskLow', 'AskClose', 'LastAskSize']

    all_history_dict = {}

    for i, pair in enumerate(pairs):
        file_name_zip = path_join(*[data_dir, market, resolution, pair + '.zip'])
        history = pd.read_csv(file_name_zip, compression='zip', delimiter=',',
                              parse_dates=True, names=column_names,
                              index_col='datetime',
                              header=None)

        history['Open'] = history[['BidOpen', 'AskOpen']].mean(axis=1)
        history['High'] = history[['BidHigh', 'AskHigh']].mean(axis=1)
        history['Low'] = history[['BidLow', 'AskLow']].mean(axis=1)
        history['Close'] = history[['BidClose', 'AskClose']].mean(axis=1)

        all_history_dict[pair] = history[['Open', 'High', 'Low', 'Close']]

    #   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

    all_history_df = pd.concat([df for df in list(all_history_dict.values())], axis=1, join='inner')

    row_count = all_history_df.shape[0]
    start_row = row_count - period - 1
    start_date = all_history_df.index.values[start_row]
    end_row = row_count - 1
    end_date = all_history_df.index.values[end_row]

    print(
        f'row_count={row_count}, start_row={start_row}, start_date={start_date}, end_row={end_row}, end_date={end_date}')

    for i, pair in enumerate(pairs):
        df_data = all_history_df.iloc[start_row:end_row, i * 4:i * 4 + 4]
        np_data = np.array(df_data)

        data = np.resize(data, (data.shape[0], np_data.shape[0], data.shape[2]))

        data[i] = np_data

        #   i += 1

    return data


def prepare_data_train_with_offset(data_dir, market, resolution, pairs, period, offset):
    data = np.empty(shape=(len(pairs), 0, 4), dtype=np.float)

    column_names = ['datetime', 'BidOpen', 'BidHigh', 'BidLow', 'BidClose', 'LastBidSize', 'AskOpen', 'AskHigh',
                    'AskLow', 'AskClose', 'LastAskSize']

    all_history_dict = {}

    for i, pair in enumerate(pairs):
        file_name_zip = path_join(*[data_dir, market, resolution, pair + '.zip'])
        history = pd.read_csv(file_name_zip, compression='zip', delimiter=',',
                              parse_dates=True, names=column_names,
                              index_col='datetime',
                              header=None)

        history['Open'] = history[['BidOpen', 'AskOpen']].mean(axis=1)
        history['High'] = history[['BidHigh', 'AskHigh']].mean(axis=1)
        history['Low'] = history[['BidLow', 'AskLow']].mean(axis=1)
        history['Close'] = history[['BidClose', 'AskClose']].mean(axis=1)

        all_history_dict[pair] = history[['Open', 'High', 'Low', 'Close']]

    #   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

    all_history_df = pd.concat([df for df in list(all_history_dict.values())], axis=1, join='inner')

    row_count = all_history_df.shape[0]
    start_row = row_count - offset - period - 1
    start_date = all_history_df.index.values[start_row]
    end_row = row_count - offset - 1
    end_date = all_history_df.index.values[end_row]

    print(
        f'row_count={row_count}, start_row={start_row}, start_date={start_date}, end_row={end_row}, end_date={end_date}')

    for i, pair in enumerate(pairs):
        df_data = all_history_df.iloc[start_row:end_row, i * 4:i * 4 + 4]
        np_data = np.array(df_data)

        data = np.resize(data, (data.shape[0], np_data.shape[0], data.shape[2]))

        data[i] = np_data

        #   i += 1

    return data


def prepare_data_train_with_dates(data_dir, market, resolution, pairs, start_date, end_date):
    data = np.empty(shape=(len(pairs), 0, 4), dtype=np.float)

    column_names = ['datetime', 'BidOpen', 'BidHigh', 'BidLow', 'BidClose', 'LastBidSize', 'AskOpen', 'AskHigh',
                    'AskLow', 'AskClose', 'LastAskSize']

    all_history_dict = {}

    for i, pair in enumerate(pairs):
        file_name_zip = path_join(*[data_dir, market, resolution, pair + '.zip'])
        history = pd.read_csv(file_name_zip, compression='zip', delimiter=',',
                              parse_dates=True, names=column_names,
                              index_col='datetime',
                              header=None)

        history['Open'] = history[['BidOpen', 'AskOpen']].mean(axis=1)
        history['High'] = history[['BidHigh', 'AskHigh']].mean(axis=1)
        history['Low'] = history[['BidLow', 'AskLow']].mean(axis=1)
        history['Close'] = history[['BidClose', 'AskClose']].mean(axis=1)

        history = history[start_date: end_date]

        all_history_dict[pair] = history[['Open', 'High', 'Low', 'Close']]

    #   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

    all_history_df = pd.concat([df for df in list(all_history_dict.values())], axis=1, join='inner')

    print(all_history_df.head())
    print(all_history_df.tail())

    # row_count = all_history_df.shape[0]
    # start_row = row_count - period - 1
    # start_date = all_history_df.index.values[start_row]
    # end_row = row_count - 1
    # end_date = all_history_df.index.values[end_row]

    # print(
    #     f'row_count={row_count}, start_row={start_row}, start_date={start_date}, end_row={end_row}, end_date={end_date}')

    # print(
    #     f'start_date={start_date}, end_date={end_date}')

    for i, pair in enumerate(pairs):
        # df_data = all_history_df.iloc[start_row:end_row, i * 4:i * 4 + 4]
        df_data = all_history_df.iloc[:, i * 4:i * 4 + 4]
        np_data = np.array(df_data)

        data = np.resize(data, (data.shape[0], np_data.shape[0], data.shape[2]))

        data[i] = np_data

        #   i += 1

    return start_date, end_date, data


def prepare_data_train_test(data_dir, market, resolution, pairs, period, train_test_split):
    data = np.empty(shape=(len(pairs), 0, 4), dtype=np.float)

    column_names = ['datetime', 'BidOpen', 'BidHigh', 'BidLow', 'BidClose', 'LastBidSize', 'AskOpen', 'AskHigh',
                    'AskLow', 'AskClose', 'LastAskSize']

    all_history_dict = {}

    for i, pair in enumerate(pairs):
        file_name_zip = path_join(*[data_dir, market, resolution, pair + '.zip'])
        history = pd.read_csv(file_name_zip, compression='zip', delimiter=',',
                              parse_dates=True, names=column_names,
                              index_col='datetime',
                              header=None)

        history['Open'] = history[['BidOpen', 'AskOpen']].mean(axis=1)
        history['High'] = history[['BidHigh', 'AskHigh']].mean(axis=1)
        history['Low'] = history[['BidLow', 'AskLow']].mean(axis=1)
        history['Close'] = history[['BidClose', 'AskClose']].mean(axis=1)

        all_history_dict[pair] = history[['Open', 'High', 'Low', 'Close']]

    #   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
    all_history_df = pd.concat([df for df in list(all_history_dict.values())], axis=1, join='inner')

    row_count = all_history_df.shape[0]
    start_row = row_count - period - 1
    start_date = all_history_df.index.values[start_row]
    end_row = int(row_count - period * (1 - train_test_split))
    end_date = all_history_df.index.values[end_row]

    print(
        f'row_count={row_count}, start_row={start_row}, start_date={start_date}, end_row={end_row}, end_date={end_date}')

    for i, pair in enumerate(pairs):
        df_data = all_history_df.iloc[start_row:end_row, i * 4:i * 4 + 4]
        np_data = np.array(df_data)

        data = np.resize(data, (data.shape[0], np_data.shape[0], data.shape[2]))

        data[i] = np_data

        i += 1

    return data


def prepare_data_test(data_dir, market, resolution, pairs, period):
    data = np.empty(shape=(len(pairs), 0, 4), dtype=np.float)

    column_names = ['datetime', 'BidOpen', 'BidHigh', 'BidLow', 'BidClose', 'LastBidSize', 'AskOpen', 'AskHigh',
                    'AskLow', 'AskClose', 'LastAskSize']

    all_history_dict = {}

    for i, pair in enumerate(pairs):
        file_name_zip = path_join(*[data_dir, market, resolution, pair + '.zip'])
        history = pd.read_csv(file_name_zip, compression='zip', delimiter=',',
                              parse_dates=True, names=column_names,
                              index_col='datetime',
                              header=None)

        history['Open'] = history[['BidOpen', 'AskOpen']].mean(axis=1)
        history['High'] = history[['BidHigh', 'AskHigh']].mean(axis=1)
        history['Low'] = history[['BidLow', 'AskLow']].mean(axis=1)
        history['Close'] = history[['BidClose', 'AskClose']].mean(axis=1)

        all_history_dict[pair] = history[['Open', 'High', 'Low', 'Close']]

    #   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

    all_history_df = pd.concat([df for df in list(all_history_dict.values())], axis=1, join='inner')

    row_count = all_history_df.shape[0]
    start_row = row_count - period - 1
    start_date = all_history_df.index.values[start_row]
    end_row = row_count - 1
    end_date = all_history_df.index.values[end_row]

    print(
        f'row_count={row_count}, start_row={start_row}, start_date={start_date}, end_row={end_row}, end_date={end_date}')

    for i, pair in enumerate(pairs):
        df_data = all_history_df.iloc[start_row:end_row, i * 4:i * 4 + 4]
        np_data = np.array(df_data)

        data = np.resize(data, (data.shape[0], np_data.shape[0], data.shape[2]))

        data[i] = np_data

        #   i += 1

    return data


def prepare_data_test_with_dates(data_dir, market, resolution, pairs, period):
    data = np.empty(shape=(len(pairs), 0, 4), dtype=np.float)

    column_names = ['datetime', 'BidOpen', 'BidHigh', 'BidLow', 'BidClose', 'LastBidSize', 'AskOpen', 'AskHigh',
                    'AskLow', 'AskClose', 'LastAskSize']

    all_history_dict = {}

    for i, pair in enumerate(pairs):
        file_name_zip = path_join(*[data_dir, market, resolution, pair + '.zip'])
        history = pd.read_csv(file_name_zip, compression='zip', delimiter=',',
                              parse_dates=True, names=column_names,
                              index_col='datetime',
                              header=None)

        history['Open'] = history[['BidOpen', 'AskOpen']].mean(axis=1)
        history['High'] = history[['BidHigh', 'AskHigh']].mean(axis=1)
        history['Low'] = history[['BidLow', 'AskLow']].mean(axis=1)
        history['Close'] = history[['BidClose', 'AskClose']].mean(axis=1)

        all_history_dict[pair] = history[['Open', 'High', 'Low', 'Close']]

    #   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
    all_history_df = pd.concat([df for df in list(all_history_dict.values())], axis=1, join='inner')

    row_count = all_history_df.shape[0]
    start_row = row_count - period
    start_date = all_history_df.index.values[start_row]
    end_row = row_count - 1
    end_date = all_history_df.index.values[end_row]

    print(
        f'row_count={row_count}, start_row={start_row}, start_date={start_date}, end_row={end_row}, end_date={end_date}')

    for i, pair in enumerate(pairs):
        df_data = all_history_df.iloc[start_row:end_row, i * 4:i * 4 + 4]
        np_data = np.array(df_data)

        data = np.resize(data, (data.shape[0], np_data.shape[0], data.shape[2]))

        data[i] = np_data

        #   i += 1

    return data


def prepare_data_test_with_end_date(data_dir, market, resolution, pairs, period, end_date):
    data = np.empty(shape=(len(pairs), 0, 4), dtype=np.float)

    column_names = ['datetime', 'BidOpen', 'BidHigh', 'BidLow', 'BidClose', 'LastBidSize', 'AskOpen', 'AskHigh',
                    'AskLow', 'AskClose', 'LastAskSize']

    all_history_dict = {}

    # print('Start collecting data...')

    for i, pair in enumerate(pairs):
        file_name_zip = path_join(*[data_dir, market, resolution, pair + '.zip'])
        history = pd.read_csv(file_name_zip, compression='zip', delimiter=',',
                              parse_dates=True, names=column_names,
                              index_col='datetime',
                              header=None)

        history['Open'] = history[['BidOpen', 'AskOpen']].mean(axis=1)
        history['High'] = history[['BidHigh', 'AskHigh']].mean(axis=1)
        history['Low'] = history[['BidLow', 'AskLow']].mean(axis=1)
        history['Close'] = history[['BidClose', 'AskClose']].mean(axis=1)

        all_history_dict[pair] = history[['Open', 'High', 'Low', 'Close']]

    #   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
    all_history_df = pd.concat([df for df in list(all_history_dict.values())], axis=1, join='inner')

    all_history_df.index = all_history_df.index.round('D')
    end_date = end_date.date()

    # if end_date.weekday() == 7:
    #     end_date -= timedelta(2)
    # elif end_date.weekday() == 6:
    #     end_date -= timedelta(1)
    #
    # print(f'end_date: {end_date}')
    # print(f'end_date_weekday: {end_date.weekday()}')

    row_count = all_history_df.shape[0]

    # print(all_history_df.head(10))
    # print(all_history_df.tail(10))

    end_row = all_history_df.index.get_loc(end_date, method='nearest')

    # print(end_row)

    end_date = all_history_df.index.values[end_row]
    start_row = end_row - period

    # print(start_row)

    start_date = all_history_df.index.values[start_row]

    # print(
    #     f'row_count={row_count}, start_row={start_row}, start_date={start_date}, end_row={end_row}, end_date={end_date}')

    print(
        f'row_count={row_count}, start_row={start_row}, start_date={start_date}, end_row={end_row}, end_date={end_date}')

    for i, pair in enumerate(pairs):
        df_data = all_history_df.iloc[start_row:end_row, i * 4:i * 4 + 4]
        np_data = np.array(df_data)

        data = np.resize(data, (data.shape[0], np_data.shape[0], data.shape[2]))

        data[i] = np_data

        #   i += 1

    return data


def prepare_data_trade_fxcm(FXCM_ACCESS_TOKEN, server, pairs, period, number):
    data = np.empty(shape=(len(pairs), 0, 4), dtype=np.float)
    conn = fxcmpy.fxcmpy(access_token=FXCM_ACCESS_TOKEN, server=server, log_level='error')

    data_dict = {}

    for i, pair in enumerate(pairs):
        history = conn.get_candles(pair, period=period, number=number)
        history.reset_index(level=0, inplace=True)

        history = history[
            ['date', 'bidopen', 'bidhigh', 'bidlow', 'bidclose', 'tickqty', 'askopen', 'askhigh', 'asklow', 'askclose',
             'tickqty']]

        history['date'] = history['date'].map(time_format_changer_fxcm)

        history['Open'] = history[['bidopen', 'askopen']].mean(axis=1)
        history['High'] = history[['bidhigh', 'askhigh']].mean(axis=1)
        history['Low'] = history[['bidlow', 'asklow']].mean(axis=1)
        history['Close'] = history[['bidclose', 'askclose']].mean(axis=1)

        data_dict[pair] = history[['Open', 'High', 'Low', 'Close']]

    #   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

    conn.close()

    data_df = pd.concat([df for df in list(data_dict.values())], axis=1, join='inner')

    row_count = data_df.shape[0]
    start_row = int(row_count) - number
    start_date = data_df.index.values[start_row]
    end_row = int(row_count) - 1
    end_date = data_df.index.values[end_row]

    print(
        f'row_count={row_count}, start_row={start_row}, start_date={start_date}, end_row={end_row}, end_date={end_date}')

    for i, pair in enumerate(pairs):
        df_data = data_df.iloc[start_row:end_row, i * 4:i * 4 + 4]
        np_data = np.array(df_data)
        data = np.resize(data, (data.shape[0], np_data.shape[0], data.shape[2]))
        data[i] = np_data

        #   i += 1

    return data


def prepare_data_trade_oanda(OANDA_ACCESS_TOKEN, pairs, period, number):
    data = np.empty(shape=(len(pairs), 0, 4), dtype=np.float)
    conn = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN)

    data_dict = {}

    params = {'count': number,
              'granularity': period,
              'price': 'M'}

    for i, pair in enumerate(pairs):
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        conn.request(r)

        mid = []
        for candle in r.response['candles']:
            mid.append([candle['time'], candle['mid']['o'], candle['mid']['h'], candle['mid']['l'], candle['mid']['c'],
                        candle['volume']])

        history = pd.DataFrame(mid)
        history.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        history = history.set_index('Time')

        history.index = history.index.map(time_format_changer_oanda)

        data_dict[pair] = history[['Open', 'High', 'Low', 'Close']]

    #   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

    data_df = pd.concat([df for df in list(data_dict.values())], axis=1, join='inner')

    row_count = data_df.shape[0]
    start_row = int(row_count) - number
    start_date = data_df.index.values[start_row]
    end_row = int(row_count) - 1
    end_date = data_df.index.values[end_row]

    print(
        f'row_count={row_count}, start_row={start_row}, start_date={start_date}, end_row={end_row}, end_date={end_date}')

    for i, pair in enumerate(pairs):
        df_data = data_df.iloc[start_row:end_row, i * 4:i * 4 + 4]
        np_data = np.array(df_data)
        data = np.resize(data, (data.shape[0], np_data.shape[0], data.shape[2]))
        data[i] = np_data

        #   i += 1

    return data


def prepare_data_optuna(data_dir, market, resolution, pairs, period, train_test_split, verbose):
    data_train = np.empty(shape=(len(pairs), 0, 4), dtype=np.float)
    data_eval = np.empty(shape=(len(pairs), 0, 4), dtype=np.float)

    column_names = ['datetime', 'BidOpen', 'BidHigh', 'BidLow', 'BidClose', 'LastBidSize', 'AskOpen', 'AskHigh',
                    'AskLow', 'AskClose', 'LastAskSize']

    all_history_dict = {}

    for i, pair in enumerate(pairs):
        file_name_zip = path_join(*[data_dir, market, resolution, pair + '.zip'])
        history = pd.read_csv(file_name_zip, compression='zip', delimiter=',',
                              parse_dates=True, names=column_names,
                              index_col='datetime',
                              header=None)

        history['Open'] = history[['BidOpen', 'AskOpen']].mean(axis=1)
        history['High'] = history[['BidHigh', 'AskHigh']].mean(axis=1)
        history['Low'] = history[['BidLow', 'AskLow']].mean(axis=1)
        history['Close'] = history[['BidClose', 'AskClose']].mean(axis=1)

        all_history_dict[pair] = history[['Open', 'High', 'Low', 'Close']]

    #   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
    all_history_df = pd.concat([df for df in list(all_history_dict.values())], axis=1, join='inner')

    #   Train data
    row_count = all_history_df.shape[0]
    start_row = row_count - period - 1
    start_date = all_history_df.index.values[start_row]
    end_row = int(row_count - period * (1 - train_test_split))
    end_date = all_history_df.index.values[end_row]

    if verbose:
        print(
            f'row_count={row_count}, train_start_row={start_row}, train_start_date={start_date}, train_end_row={end_row}, train_end_date={end_date}')

    for i, pair in enumerate(pairs):
        df_data = all_history_df.iloc[start_row:end_row, i * 4:i * 4 + 4]
        np_data = np.array(df_data)

        data_train = np.resize(data_train, (data_train.shape[0], np_data.shape[0], data_train.shape[2]))

        data_train[i] = np_data

        i += 1

    #   Eval data
    row_count = all_history_df.shape[0]
    start_row = int(row_count - period * (1 - train_test_split)) + 1
    start_date = all_history_df.index.values[start_row]
    end_row = row_count - 1
    end_date = all_history_df.index.values[end_row]

    if verbose:
        print(
            f'row_count={row_count}, eval_start_row={start_row}, eval_start_date={start_date}, eval_end_row={end_row}, eval_end_date={end_date}')

    for i, pair in enumerate(pairs):
        df_data = all_history_df.iloc[start_row:end_row, i * 4:i * 4 + 4]
        np_data = np.array(df_data)

        data_eval = np.resize(data_eval, (data_eval.shape[0], np_data.shape[0], data_eval.shape[2]))

        data_eval[i] = np_data

        i += 1

    return data_train, data_eval


def calculate_features(data, compute_indicators):
    v_features_data = np.zeros((0, 0, 0), dtype=np.float32)

    for i in range(data.shape[0]):
        security = pd.DataFrame(data[i, :, :]).fillna(method='ffill').fillna(method='bfill')
        security.columns = ['Open', 'High', 'Low', 'Close']

        v_technical_features = technical_features(security=security.astype(float), open_name='Open',
                                                  high_name='High', low_name='Low', close_name='Close')

        if compute_indicators == 'prices':
            v_feature_data = np.asarray(v_technical_features.get_indicators_prices())
        elif compute_indicators == 'returns':
            v_feature_data = np.asarray(v_technical_features.get_indicators_returns())
        elif compute_indicators == 'log_returns':
            v_feature_data = np.asarray(v_technical_features.get_indicators_log_returns())
        elif compute_indicators == 'returns_hlc':
            v_feature_data = np.asarray(v_technical_features.get_indicators_returns_hlc())
        elif compute_indicators == 'log_returns_hlc':
            v_feature_data = np.asarray(v_technical_features.get_indicators_log_returns_hlc())
        elif compute_indicators == 'patterns':
            v_feature_data = np.asarray(v_technical_features.get_indicators_patterns())
        elif compute_indicators == 'returns_patterns_volatility':
            v_feature_data = np.asarray(v_technical_features.get_indicators_returns_patterns_volatility())
        elif compute_indicators == 'momentum_simple':
            v_feature_data = np.asarray(v_technical_features.get_indicators_momentum('simple'))
        elif compute_indicators == 'momentum_multi':
            v_feature_data = np.asarray(v_technical_features.get_indicators_momentum('multi'))
        elif compute_indicators == 'all':
            v_feature_data = np.asarray(v_technical_features.get_indicators_all())
        elif compute_indicators == 'misc':
            v_feature_data = np.asarray(v_technical_features.get_indicators_misc())
        elif compute_indicators == 'random':
            v_feature_data = np.asarray(v_technical_features.get_indicators_random())

        v_features_data = np.resize(v_features_data,
                                    (v_features_data.shape[0] + 1, v_feature_data.shape[0], v_feature_data.shape[1]))
        v_features_data[i] = v_feature_data

        # price_data = new_data[:, :, :4]
        # features_data = new_data[:, :, 4:]

    return v_features_data


# def calculate_features_new(data, compute_indicators):
#     v_features_data = np.zeros((0, 0, 0), dtype=np.float32)
#
#     for i in range(data.shape[0]):
#         security = pd.DataFrame(data[i, :, :]).fillna(method='ffill').fillna(method='bfill')
#         security['Volume'] = 1
#         security.columns = ['open', 'high', 'low', 'close', 'volume']
#         # security.ta.cores = 6
#         # security.ta.strategy(compute_indicators, exclude=["vwap"] + ta.Category["volume"], timed=False)
#         security.ta.strategy(compute_indicators, exclude=["vwap"], timed=False)
#         security.fillna(0, inplace=True)
#
#         v_features_data = np.resize(v_features_data,
#                                     (v_features_data.shape[0] + 1, security.shape[0], security.shape[1]))
#         v_features_data[i] = security
#
#     return v_features_data


def get_volatility_clusters(data_dir, market, resolution, pairs, period):
    v_data = prepare_data_cluster(data_dir, market, resolution, pairs, period)

    #   print(v_data.head())

    # making a copy of original data
    v_data_copy = v_data.copy()

    # Droping 'Date' feature from data
    v_data_copy = v_data_copy.drop('Date', axis=1)

    # Calculating the annual Returns for each currency pairs in our dataset
    Calc = v_data_copy.pct_change().mean() * 252
    Calc = pd.DataFrame(Calc)
    Calc.columns = ['Returns']

    # Calculating the Volatility (standard deviation) for each currency pairs in our dataset
    Calc['Volatility'] = v_data_copy.pct_change().std() * sqrt(252)

    # Converting the format to numpy array
    Calc_Conv = np.asarray([np.asarray(Calc['Returns']), np.asarray(Calc['Volatility'])]).T

    # Silhoutte analysis to find the optimum claster numbers between 6, 7, 8, 9 and 10 clusters (based on Elbow Curve)
    range_of_k = list(range(5, 11))
    Mean_Silhouette_Scores = []

    for k in range_of_k:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(15, 7)

        km = KMeans(n_clusters=k)
        labels = km.fit_predict(Calc_Conv)
        centroids = km.cluster_centers_

        Silhouette_Smpls = silhouette_samples(Calc_Conv, labels)

        # Plotting Silhouette
        y_Lband = 0
        y_Uband = 0
        for i, cluster in enumerate(np.unique(labels)):
            cluster_Silhouette_Smpls = Silhouette_Smpls[labels == cluster]
            cluster_Silhouette_Smpls.sort()
            y_Uband += len(cluster_Silhouette_Smpls)
            ax.barh(range(y_Lband, y_Uband), cluster_Silhouette_Smpls, edgecolor='none', height=1)
            ax.text(-0.03, (y_Lband + y_Uband) / 2, str(i + 1))
            y_Lband += len(cluster_Silhouette_Smpls)

        # Calculating and Plotting the average silhouette score
        Avg_Silhouette_Score = np.mean(Silhouette_Smpls)
        Mean_Silhouette_Scores.append(Avg_Silhouette_Score)
        ax.axvline(Avg_Silhouette_Score, linestyle='-.', linewidth=2, color='red')
        ax.set_xlim([-0.1, 1])
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Clusters')
        plt.tight_layout()

        plt.suptitle(f'Silhouette analysis using k = {k}', fontsize=16, fontweight='semibold', y=1.02)

    # Finding the optimum cluster number based on the max Average of Silhouette Coefficients
    Optimum_Clusters = range_of_k[Mean_Silhouette_Scores.index(max(Mean_Silhouette_Scores))]

    # Based on results of Elbow Curve and Silhouette Analysis, the optimum number of clusters should be Optimum_Clusters.
    km = KMeans(n_clusters=Optimum_Clusters)
    labels = km.fit_predict(Calc_Conv)
    centroids = km.cluster_centers_

    # Plotting the clusters and their centroids
    fig, ax = plt.subplots(figsize=(15, 7))
    for i in range(Optimum_Clusters):
        plt.scatter(Calc_Conv[km.labels_ == i, 0], Calc_Conv[km.labels_ == i, 1], label='cluster ' + str(i + 1))

    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=100, c='r', label='centroids')
    plt.legend()
    plt.title('Clustered Volatilities', fontweight='bold')

    Cluster_details = [(name, cluster) for name, cluster in zip(Calc.index, labels)]

    clusters = []
    for i in range(0, Optimum_Clusters):
        # for i in range (0, 4):
        tmp = []
        for detail in Cluster_details:
            if detail[1] == i:
                tmp.append(detail[0])
        clusters.append(tmp)

    volatility_clusters = {}
    for i in range(len(clusters)):
        #   print('Cluster ', i + 1, '= ', clusters[i])
        volatility_clusters['cluster_' + str(i + 1)] = clusters[i]

    return volatility_clusters
