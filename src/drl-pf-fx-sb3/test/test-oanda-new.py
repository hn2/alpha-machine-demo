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
import pandas as pd
import multiprocessing
from collections import deque
from datetime import timedelta
from distutils.util import strtobool

from findrl.forex_utils import get_forex_7, get_forex_12, get_forex_18, get_forex_28
from findrl.model_utils import parse_model_name
from findrl.run_utils import run_model_make_env_new

#   cd util_lib
#   python -m tensorboard.main --logdir=.
#   tensorboard --logdir=.

DELIMITER = '-'
ENV_VERBOSE = True

MODELS_DIR = r'C:\alpha-machine\models\forex\oanda\daily\new'
MODEL_NAME = 'fx_sb3_leverage_20_train_with_callback_with_random_episode_start_noise_none-7-100100-1000-30-2-oanda-daily-on_algo.a2c-comp_pos.long_and_short-comp_ind.all-comp_rew.[log_returns]-true-2e74e5f9'
LEVERAGE = 20


class main(QCAlgorithm):

    def Initialize(self):
        self.number_of_instruments, self.market_name, self.resolution_name, self.env_lookback_period, self.spread, self.online_algorithm, self.compute_position, self.compute_indicators, self.compute_reward, self.meta_rl = parse_model_name(
            MODEL_NAME, DELIMITER)

        self.meta_rl = strtobool(self.meta_rl)

        self.Debug(self.number_of_instruments)
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

        if int(self.number_of_instruments) == 7:
            self.instruments, self.pip_size, self.pip_spread = get_forex_7(int(self.spread))
        elif int(self.number_of_instruments) == 12:
            self.instruments, self.pip_size, self.pip_spread = get_forex_12(int(self.spread))
        elif int(self.number_of_instruments) == 18:
            self.instruments, self.pip_size, self.pip_spread = get_forex_18(int(self.spread))
        elif int(self.number_of_instruments) == 28:
            self.instruments, self.pip_size, self.pip_spread = get_forex_28(int(self.spread))

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
        self.SetStartDate(2021, 11, 18)
        self.SetEndDate(2022, 2, 26)

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

        '''
        now = datetime.now()
        self.SetStartDate(2021, 11, 18)
        self.SetEndDate(2022, 2, 26)
        '''

        # online model
        self.online_model = None
        self.deterministic = True
        self.verbose = 1

        self.rollingwindow = {}
        symbols = self.Securities.Keys
        history = self.History(symbols, self.trade_lookback_period, self.resolution)
        for symbol in symbols:
            self.rollingwindow[symbol] = QuoteBarData(self, symbol, self.trade_lookback_period, history.loc[symbol],
                                                      self.resolution)

        #   self.Schedule.On(self.DateRules.EveryDay("EURUSD"), self.TimeRules.BeforeMarketClose("EURUSD", 5),
        #                    self.LiquidateAtClose)

    def OnData(self, data):
        '''
        for key in data.Keys:
            self.Log(str(key.Value) + ": " + str(data.Time) + " > " + str(data[key].Value))
        '''
        for _, QuoteBarData in self.rollingwindow.items():
            self.Debug("QuoteBarData")

        self.rebalance()

    def LiquidateAtClose(self):
        self.Liquidate()

    def rebalance(self):
        for symbol, QuoteBarData in self.rollingwindow.items():

            open = np.asarray(QuoteBarData.open)
            if open.shape[0] < self.trade_lookback_period:
                return

        self.Debug('Running model...')

        v_data, v_features = self.prepare_data_trade()
        self.Debug(f'compute_indicators: {self.compute_indicators}')
        self.Debug(f'Data shape:{np.shape(v_data)}, Features shape:{np.shape(v_features)}')

        v_suggested_weights, v_suggested_positions = run_model_make_env_new(MODELS_DIR,
                                                                            MODEL_NAME,
                                                                            DELIMITER,
                                                                            self.deterministic,
                                                                            v_data,
                                                                            v_features,
                                                                            self.instruments,
                                                                            self.env_lookback_period,
                                                                            False,
                                                                            self.cash,
                                                                            self.max_slippage_percent,
                                                                            self.lot_size,
                                                                            self.leverage,
                                                                            self.pip_size,
                                                                            self.pip_spread,
                                                                            self.compute_position,
                                                                            self.compute_reward,  # returns log_returns
                                                                            self.meta_rl,
                                                                            ENV_VERBOSE)

        self.Debug(self.Time)
        self.Debug(self.UtcTime)
        self.Debug(f"Date:{self.Time.date()}")
        self.Debug(self.instruments)
        self.Debug(f'Suggested weights: {v_suggested_weights}')
        self.Debug(f'Suggested positions: {v_suggested_positions}')
        self.Debug(f"Before rebalance:{self.Portfolio.TotalPortfolioValue}")

        #   self.Liquidate()

        for i, instrument in enumerate(self.instruments):
            self.SetHoldings(instrument, self.leverage * v_suggested_weights[i])

        self.Debug(f"After rebalance:{self.Portfolio.TotalPortfolioValue}")
        self.Debug("================")

    # def prepare_data_trade(self):
    #     data = np.empty(shape=(len(self.instruments), 0, 4), dtype=np.float)
    #
    #     i = 0
    #
    #     for instrument in self.instruments:
    #         history = self.History(self.Symbol(instrument), self.trade_lookback_period, self.resolution)
    #
    #         df_data = history[['open', 'high', 'low', 'close']]
    #         np_data = np.array(df_data)
    #
    #         data = np.resize(data, (data.shape[0], np_data.shape[0], data.shape[2]))
    #
    #         data[i] = np_data
    #
    #         i += 1
    #
    #     return data

    def prepare_data_trade(self):
        #   data = np.empty(shape=(len(self.instruments), 0, 4), dtype=np.float)
        data = np.zeros((0, 0, 0), dtype=np.float)
        features = np.zeros((0, 0, 0), dtype=np.float)

        i = 0

        for instrument in self.instruments:
            history = self.History(self.Symbol(instrument), self.trade_lookback_period, self.resolution)

            security = history[['open', 'high', 'low', 'close']]
            security['volume'] = 1
            np_data = np.array(security)
            # security.ta.strategy(self.compute_indicators, exclude=["vwap"], timed=False)
            #   security.ta.strategy(self.compute_indicators, exclude=["vwap"], timed=False)
            security.fillna(0, inplace=True)
            np_features = np.array(security)

            data = np.resize(data, (data.shape[0] + 1, np_data.shape[0], np_data.shape[1]))
            features = np.resize(features, (features.shape[0] + 1, np_features.shape[0], np_features.shape[1]))

            data[i] = np_data
            features[i] = np_features

            i += 1

        return data, features

    def calculate_features(self, data, compute_indicators):
        v_features_data = np.zeros((0, 0, 0), dtype=np.float32)

        self.Debug(f'compute_indicators: {compute_indicators}')

        for i in range(data.shape[0]):
            security = pd.DataFrame(data[i, :, :]).fillna(method='ffill').fillna(method='bfill')
            security['Volume'] = 1
            security.columns = ['open', 'high', 'low', 'close', 'volume']
            security.ta.strategy('all', exclude=["vwap"], timed=False)
            security.fillna(0, inplace=True)

            v_features_data = np.resize(v_features_data,
                                        (v_features_data.shape[0] + 1, security.shape[0], security.shape[1]))
            v_features_data[i] = security

        return v_features_data


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
