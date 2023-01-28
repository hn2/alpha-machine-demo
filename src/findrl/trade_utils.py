import fxcmpy
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd


def _time_format_changer_fxcm(date):
    return date.strftime('%Y%m%d %H:%M')


def _time_format_changer_oanda(date):
    return date.replace('-', '').replace('.000000000Z', '').replace('T', ' ')[:-3]


def prepare_data_trade_fxcm(fxcm_access_token, server, pairs, period, number):
    data = np.empty(shape=(len(pairs), 0, 4), dtype=np.float)
    conn = fxcmpy.fxcmpy(access_token=fxcm_access_token, server=server, log_level='error')

    data_dict = {}

    for i, pair in enumerate(pairs):
        history = conn.get_candles(pair, period=period, number=number)
        history.reset_index(level=0, inplace=True)

        history = history[
            ['date', 'bidopen', 'bidhigh', 'bidlow', 'bidclose', 'tickqty', 'askopen', 'askhigh', 'asklow', 'askclose',
             'tickqty']]

        history['date'] = history['date'].map(_time_format_changer_fxcm)

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


def prepare_data_trade_oanda(oanda_access_token, pairs, period, number):
    data = np.empty(shape=(len(pairs), 0, 4), dtype=np.float)
    conn = oandapyV20.API(access_token=oanda_access_token)

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

        history.index = history.index.map(_time_format_changer_oanda)

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
