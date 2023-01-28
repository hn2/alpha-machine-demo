#   https://github.com/alpha-machine/Lean/blob/master/data/forex/readme.md
#   https://realpython.com/python-f-strings/

import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd


def time_format_changer_fxcm(date):
    return date.strftime('%Y%m%d %H:%M')


def time_format_changer_oanda(date):
    return date.replace('-', '').replace('.000000000Z', '').replace('T', ' ')[:-3]


def download_data_oanda(OANDA_ACCESS_TOKEN, pairs, period, number):
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
