from clr import AddReference

AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Indicators")
AddReference("QuantConnect.Common")

from QuantConnect import *
from QuantConnect.Data import *
from QuantConnect.Algorithm import *
from QuantConnect.Brokerages import *
from QuantConnect.Data.Consolidators import *

import numpy as np
import multiprocessing
from os.path import join as path_join
from collections import deque
from datetime import timedelta
from distutils.util import strtobool

from findrl.model_utils import parse_model_name
from findrl.run_utils import run_model_make_env_sb3_qc
from findrl.file_utils import load_dict_from_pickle

#   cd util_lib
#   python -m tensorboard.main --logdir=.
#   tensorboard --logdir=.

DELIMITER = "-"
MODELS_DIR = "E:/alpha-machine-files/models/forex/oanda/daily/sb3-train"
MODEL_NAME = "fx_wk.49_lr.0.00005_b.32_ns.none_cb.True_res.True_lev.20-7-100100-1500_0-30-2-oanda-daily-on_algo.ddpg-c_pos.long_and_short-c_ind.all-c_rew.log_returns-m_rl.True-b373df2d"
LEVERAGE = 20


class main(QCAlgorithm):

    def Initialize(self):
        self.number_of_instruments, self.total_timesteps, self.market_name, self.resolution_name, self.env_lookback_period, self.spread, self.online_algorithm, self.compute_position, self.compute_indicators, self.compute_reward, self.meta_rl = parse_model_name(
            MODEL_NAME, DELIMITER)

        self.meta_rl = strtobool(self.meta_rl)

        self.Debug(self.number_of_instruments)
        self.Debug(self.total_timesteps)
        self.Debug(self.market_name)
        self.Debug(self.resolution_name)
        self.Debug(self.env_lookback_period)
        self.Debug(self.spread)
        self.Debug(self.online_algorithm)
        self.Debug(self.compute_position)
        self.Debug(self.compute_indicators)
        self.Debug(self.compute_reward)
        self.Debug(self.meta_rl)
        self.Debug(type(self.meta_rl))

        env_attributes = load_dict_from_pickle(path_join(*[MODELS_DIR, MODEL_NAME, 'online', 'env.pkl']))
        kwargs = env_attributes['kwargs']
        self.instruments = kwargs['INSTRUMENTS '].replace('INSTRUMENTS = ', '')

        # instruments_attributes = load_objects_from_pickle(path_join(*[MODELS_DIR, MODEL_NAME, 'online', 'instruments.pkl']))
        # self.account_currency = instruments_attributes[0]
        # self.instruments = instruments_attributes[1]
        # self.pip_size = instruments_attributes[2]
        # self.pip_spread = instruments_attributes[3]

        if self.market_name == 'oanda':
            self.market = Market.Oanda
        elif self.market_name == 'fxcm':
            self.market = Market.FXCM

        if self.resolution_name == 'daily' or self.resolution_name == 'day':
            self.resolution = Resolution.Daily
        elif self.resolution_name == 'hour':
            self.resolution = Resolution.Hour

        self.symbols = [self.AddForex(instrument, self.resolution, self.market).Symbol for instrument in
                        self.instruments]
        self.SetBrokerageModel(BrokerageName.OandaBrokerage)

        #   self.SetStartDate(2021, 8, 14)
        self.SetStartDate(2019, 12, 16)
        self.SetEndDate(2022, 9, 11)

        self.Debug("--------------")
        self.Debug(self.instruments)
        self.Debug(multiprocessing.get_all_start_methods())
        self.Debug(multiprocessing.cpu_count())
        self.Debug("--------------")

        self.lot_size = 'Micro'
        self.leverage = LEVERAGE
        self.max_slippage_percent = 1e-2
        #   self.max_slippage_percent = 0

        self.cash = 1e3
        self.SetCash(self.cash)

        self.trade_lookback_period = self.env_lookback_period + 100

        self.Debug(self.env_lookback_period)
        self.Debug(self.trade_lookback_period)

        self.deterministic = True
        self.env_verbose = False

        self.rollingwindow = {}
        symbols = self.Securities.Keys
        history = self.History(symbols, self.trade_lookback_period, self.resolution)
        for symbol in symbols:
            self.rollingwindow[symbol] = QuoteBarData(self, symbol, self.trade_lookback_period, history.loc[symbol],
                                                      self.resolution)

        symbols_in_portfolio = [x.Value for x in self.Portfolio.Keys]

        if self.resolution == Resolution.Daily:
            self.Schedule.On(self.DateRules.EveryDay(symbols_in_portfolio[0]), self.TimeRules.At(0, 0), self.rebalance)
        elif self.resolution == Resolution.Hour:
            self.Schedule.On(self.DateRules.EveryDay(symbols_in_portfolio[0]),
                             self.TimeRules.Every(timedelta(minutes=60)), self.rebalance)
        elif self.resolution == Resolution.Minute:
            self.Schedule.On(self.DateRules.EveryDay(symbols_in_portfolio[0]),
                             self.TimeRules.Every(timedelta(minutes=1)), self.rebalance)

    def OnData(self, data):
        '''
        for key in data.Keys:
            self.Log(str(key.Value) + ": " + str(data.Time) + " > " + str(data[key].Value))
        '''
        # for _, QuoteBarData in self.rollingwindow.items():
        #     self.Debug("QuoteBarData")
        #
        # self.rebalance()

    def LiquidateAtClose(self):
        self.Liquidate()

    def rebalance(self):
        for symbol, QuoteBarData in self.rollingwindow.items():

            open = np.asarray(QuoteBarData.open)
            if open.shape[0] < self.trade_lookback_period:
                return

        self.Debug('Running model...')

        data = self.prepare_data_trade()

        v_suggested_weights, v_suggested_positions = run_model_make_env_sb3_qc(MODELS_DIR,
                                                                               MODEL_NAME,
                                                                               DELIMITER,
                                                                               self.deterministic,
                                                                               data,
                                                                               self.start_date,
                                                                               self.end_date,
                                                                               self.number_of_instruments,
                                                                               False,
                                                                               self.env_lookback_period,
                                                                               self.args,
                                                                               self.kwargs,
                                                                               self.compute_position,
                                                                               self.compute_indicators,
                                                                               self.compute_reward,
                                                                               # returns log_returns
                                                                               self.meta_rl,
                                                                               self.env_verbose)

        self.Debug(self.Time)
        self.Debug(self.UtcTime)
        self.Debug(f"Date:{self.Time.date()}")
        self.Debug(self.instruments)
        self.Debug(f'Suggested weights: {v_suggested_weights}')
        self.Debug(f'Suggested positions: {v_suggested_positions}')
        self.Debug(f"Before rebalance:{self.Portfolio.TotalPortfolioValue}")

        self.Debug('Invested')
        Invested = [x.Symbol.Value for x in self.Portfolio.Values if x.Invested]
        for x in Invested:
            self.Debug(x)

        self.Debug('Instruments')
        Instruments = [x.upper() for x in self.instruments]
        for x in Instruments:
            self.Debug(x)

        symbols_to_liquidate = [x for x in Invested if x not in Instruments]

        self.Debug('symbols_to_liquidate')
        for x in symbols_to_liquidate:
            self.Debug(x)

        for x in symbols_to_liquidate:
            self.Debug(f'Liquidating {str(x)} ...')
            self.Liquidate(str(x))

        for i, instrument in enumerate(self.instruments):
            self.SetHoldings(instrument, self.leverage * v_suggested_weights[i])

        self.Debug(f"After rebalance:{self.Portfolio.TotalPortfolioValue}")
        self.Debug("================")

    def prepare_data_trade(self):
        data = np.empty(shape=(len(self.instruments), 0, 4), dtype=np.float)

        i = 0

        for instrument in self.instruments:
            history = self.History(self.Symbol(instrument), self.trade_lookback_period, self.resolution)

            df_data = history[['open', 'high', 'low', 'close']]
            np_data = np.array(df_data)

            data = np.resize(data, (data.shape[0], np_data.shape[0], data.shape[2]))

            data[i] = np_data

            i += 1

        return data


class QuoteBarData:

    def __init__(self, algorithm, symbol, lookback, history, resolution):
        self.open = deque(maxlen=lookback)
        self.high = deque(maxlen=lookback)
        self.low = deque(maxlen=lookback)
        self.close = deque(maxlen=lookback)
        self.Debug = algorithm.Debug

        self.time = algorithm.Time

        if resolution == Resolution.Daily:
            consolidator = QuoteBarConsolidator(timedelta(days=1))
        elif resolution == Resolution.Hour:
            consolidator = QuoteBarConsolidator(timedelta(hours=1))
        elif resolution == Resolution.Minute:
            consolidator = QuoteBarConsolidator(timedelta(minutes=1))

        algorithm.SubscriptionManager.AddConsolidator(symbol, consolidator)
        consolidator.DataConsolidated += self.OnDataConsolidated

        for time, row in history.iterrows():
            self.time = time
            self.Update(time, row.open, row.high, row.low, row.close)

    def Update(self, time, o, h, l, c):
        self.time = time
        self.open.append(o)
        self.high.append(h)
        self.low.append(l)
        self.close.append(c)

    def OnDataConsolidated(self, sender, bar):
        self.Update(bar.EndTime, bar.Open, bar.High, bar.Low, bar.Close)
