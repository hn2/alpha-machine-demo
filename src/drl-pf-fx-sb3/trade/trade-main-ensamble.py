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

from findrl.config_utils import get_trade_params
from findrl.model_utils import parse_model_name, get_models_and_stats_lists_from_position
from findrl.file_utils import load_dict_from_pickle
from findrl.run_utils import run_model_make_env_sb3
from findrl.math_utils import math_utils
from findrl.general_utils import convert_lists_to_dict, upload_to_dropbox
from findrl.fxcm_utils import delete_pending_orders, liquidate, rebalance
from findrl.forex_utils import get_forex_28


class main(QCAlgorithm):

    def Initialize(self):
        self.algorithms, self.sort_columns, self.stat_column, self.position, self.number, self.trade_oanda, self.rebalance_oanda, self.liquidate_oanda, self.trade_fxcm, self.delete_pending_orders_fxcm, self.rebalance_fxcm, self.liquidate_fxcm, self.upload_dropbox, self.dropbox_file_name, self.dropbox_remote_dir = get_trade_params(
            CONFIG_FILE)

        # self.fxcm_access_token_real_1, self.fxcm_access_token_real_2, self.fxcm_access_token_demo_1, self.fxcm_access_token_demo_2, self.fxcm_access_token_demo_3, self.fxcm_access_token_demo_4, self.fxcm_access_token_demo_5, self.oanda_access_token, self.dropbox_access_token, self.github_access_token, self.aws_server_public_key, self.aws_server_secret_key = get_tokens_params(
        #     CONFIG_FILE)

        if STATS_FILE is None:
            self.model_names = get_models(MODELS_DIR, FILES_LOOKBACK_HOURS, INCLUDE_PATTERNS)
        else:
            self.params = {'algos': self.algorithms, 'sort_columns': self.sort_columns, 'stat_column': self.stat_column,
                           'position': self.position, 'number': self.number}

            self.model_names, self.stats = get_models_and_stats_lists_from_position(stats_file=STATS_FILE,
                                                                                    params=self.params)

        self.Debug(self.model_names)
        self.Debug(self.stats)

        self.number_of_instruments, self.total_timesteps, self.market_name, self.resolution_name, self.env_lookback_period, self.spread, self.online_algorithm, self.compute_position, self.compute_indicators, self.compute_reward, self.meta_rl = parse_model_name(
            self.model_names[0], DELIMITER)

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

        env_attributes = load_dict_from_pickle(path_join(*[MODELS_DIR, self.model_names[0], 'online', 'env.pkl']))
        self.account_currency = env_attributes.account_currency
        #   self.instruments = env_attributes.instruments
        #   self.pip_size = env_attributes.pip_size
        #   self.pip_spread = env_attributes.pip_spread

        self.instruments, self.pip_size, self.pip_spread = get_forex_28(2)

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
        self.SetBrokerageModel(BrokerageName.OandaBrokerage, AccountType.Margin)

        self.SetStartDate(2021, 8, 15)
        self.SetEndDate(2021, 11, 23)

        # self.SetStartDate(2018, 8, 1)
        # self.SetStartDate(2018, 9, 1)

        self.Debug("--------------")
        self.Debug(self.instruments)
        self.Debug(multiprocessing.get_all_start_methods())
        self.Debug(multiprocessing.cpu_count())
        self.Debug("--------------")

        self.lot_size = 'Micro'
        self.leverage = 20.0
        self.max_slippage_percent = 1e-2
        #   self.max_slippage_percent = 0

        self.cash = 1e3
        self.SetCash(self.cash)

        self.trade_lookback_period = self.env_lookback_period + 100

        self.Debug(self.env_lookback_period)
        self.Debug(self.trade_lookback_period)

        '''
        now = datetime.now()
        self.SetStartDate(2021, 4, 7)
        self.SetEndDate(2021, 7, 16)
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

        #   self.SetWarmup(self.trade_lookback_period, Resolution.Daily)
        symbols_in_portfolio = [x.Value for x in self.Portfolio.Keys]

        if self.trade_oanda:
            if self.resolution == Resolution.Daily:
                self.Schedule.On(self.DateRules.EveryDay(symbols_in_portfolio[0]), self.TimeRules.At(0, 0),
                                 self.rebalance)
            elif self.resolution == Resolution.Hour:
                self.Schedule.On(self.DateRules.EveryDay(symbols_in_portfolio[0]),
                                 self.TimeRules.Every(timedelta(minutes=60)), self.rebalance)
            elif self.resolution == Resolution.Minute:
                self.Schedule.On(self.DateRules.EveryDay(symbols_in_portfolio[0]),
                                 self.TimeRules.Every(timedelta(minutes=1)), self.rebalance)

    def OnData(self, data):
        pass

    def rebalance(self):
        for symbol, QuoteBarData in self.rollingwindow.items():

            open = np.asarray(QuoteBarData.open)
            if open.shape[0] < self.trade_lookback_period:
                return

        self.Debug('Running model...')

        v_suggested_weights_all = np.zeros(shape=len(self.instruments) + 1, dtype=np.float)
        v_suggested_weights_dict_all = convert_lists_to_dict(self.instruments, v_suggested_weights_all)
        number_of_models = len(self.model_names)

        # self.Debug(self.instruments)
        # self.Debug(v_suggested_weights_all)
        # self.Debug(v_suggested_weights_dict_all)

        for i, model_name in enumerate(self.model_names):
            number_of_instruments, total_timesteps, market_name, resolution_name, env_lookback_period, spread, online_algorithm, compute_position, compute_indicators, compute_reward, meta_rl = parse_model_name(
                model_name, DELIMITER)

            meta_rl = strtobool(meta_rl)

            env_attributes = load_dict_from_pickle(path_join(*[MODELS_DIR, model_name, 'online', 'env.pkl']))
            account_currency = env_attributes.account_currency
            self.current_instruments = env_attributes.instruments
            pip_size = env_attributes.pip_size
            pip_spread = env_attributes.pip_spread

            # self.Debug(self.current_instruments)
            # self.Debug(pip_size)
            # self.Debug(pip_spread)

            data = self.prepare_data_trade()

            v_suggested_weights, v_suggested_positions = run_model_make_env_sb3(MODELS_DIR,
                                                                                model_name,
                                                                                DELIMITER,
                                                                                self.deterministic,
                                                                                data,
                                                                                account_currency,
                                                                                self.current_instruments,
                                                                                self.env_lookback_period,
                                                                                False,
                                                                                self.cash,
                                                                                self.max_slippage_percent,
                                                                                self.lot_size,
                                                                                self.leverage,
                                                                                pip_size,
                                                                                pip_spread,
                                                                                self.compute_position,
                                                                                self.compute_indicators,
                                                                                self.compute_reward,
                                                                                # returns log_returns
                                                                                meta_rl,
                                                                                True)

            v_suggested_weights_dict = convert_lists_to_dict(self.current_instruments, v_suggested_weights)

            for key, value in v_suggested_weights_dict.items():
                v_suggested_weights_dict_all[key] += v_suggested_weights_dict.get(key)

        #   v_weights = np.average(v_suggested_weights_all, axis=0, weights=self.weights)
        v_weights = list(v_suggested_weights_dict_all.values())
        v_weights = [float(x) / number_of_models for x in v_weights]
        v_normelized_weights = math_utils.get_normelize_weights(v_weights)

        self.Debug(self.Time)
        self.Debug(self.UtcTime)
        self.Debug(f"Date:{self.Time.date()}")
        self.Debug(self.instruments)
        self.Debug(f'Normelized weights: {v_normelized_weights}')

        if self.upload_dropbox:

            self.Debug("Uploading to dropbox...")

            #   self.Debug(f"Dropbox remote dir: {self.dropbox_remote_dir}")

            try:
                upload_to_dropbox(DROPBOX_ACCESS_TOKEN, self.dropbox_file_name, DROPBOX_LOCAL_DIR,
                                  self.dropbox_remote_dir, self.instruments, v_normelized_weights)
            except Exception as e:
                self.Debug(e)

        if self.rebalance_oanda:

            self.Debug(f"Before rebalance oanda:{self.Portfolio.TotalPortfolioValue}")

            if self.liquidate_oanda:
                self.Debug('Invested oanda')
                Invested = [x.Symbol.Value for x in self.Portfolio.Values if x.Invested]
                for x in Invested:
                    self.Debug(x)

                self.Debug('Instruments oanda')
                Instruments = [x.upper() for x in self.instruments]
                for x in Instruments:
                    self.Debug(x)

                symbols_to_liquidate = [x for x in Invested if x not in Instruments]

                self.Debug('Symbols to liquidate oanda')
                for x in symbols_to_liquidate:
                    self.Debug(x)

                for x in symbols_to_liquidate:
                    self.Debug(f'Liquidating oanda {str(x)} ...')
                    self.SetHoldings(x, 0)
                    #   self.Liquidate(str(x))

            for i, instrument in enumerate(self.instruments):
                self.SetHoldings(instrument, self.leverage * v_normelized_weights[i])

            self.Debug(f"After rebalance oanda:{self.Portfolio.TotalPortfolioValue}")

        if self.trade_fxcm:

            fxcm_instruments_to_rebalance = [instrument[:3].upper() + '/' + instrument[3:].upper() for instrument in
                                             self.instruments]

            for FXCM_SERVER, FXCM_ACCESS_TOKEN, ORDER_TYPE in zip(FXCM_SERVERS, FXCM_ACCESS_TOKENS,
                                                                  ORDER_TYPES):

                self.Debug("Deleting pending orders fxcm...")

                try:

                    delete_pending_orders(FXCM_ACCESS_TOKEN, FXCM_SERVER, self.delete_pending_orders_fxcm)

                except Exception as e:
                    self.Debug(e)

                self.Debug("Liquidating fxcm...")

                try:

                    liquidate(FXCM_ACCESS_TOKEN, FXCM_SERVER, fxcm_instruments_to_rebalance, self.liquidate_fxcm)

                except Exception as e:
                    self.Debug(e)

                self.Debug("Rebalancing fxcm...")

                self.Debug(FXCM_ACCESS_TOKEN)
                self.Debug(FXCM_SERVER)
                self.Debug(ORDER_TYPE)

                try:

                    current_prices, current_pip_value_in_account_currency, current_margins_per_lot, target_positions_prices, \
                    target_positions_pip_value_in_account_currency, target_positions_margins_per_lot, current_positions, \
                    target_positions, trade_amount = rebalance(FXCM_ACCESS_TOKEN, FXCM_SERVER,
                                                               fxcm_instruments_to_rebalance,
                                                               v_normelized_weights,
                                                               self.lot_size, self.leverage,
                                                               self.account_currency, self.rebalance_fxcm,
                                                               ORDER_TYPE)

                    self.Debug(f'current_prices: {current_prices}')
                    self.Debug(f'current_pip_value_in_account_currency: {current_pip_value_in_account_currency}')
                    self.Debug(f'current_margins_per_lot: {current_margins_per_lot}')
                    self.Debug(f'target_positions_prices: {target_positions_prices}')
                    self.Debug(
                        f'target_positions_pip_value_in_account_currency: {target_positions_pip_value_in_account_currency}')
                    self.Debug(f'target_positions_margins_per_lot: {target_positions_margins_per_lot}')
                    self.Debug(f'Target positions: {target_positions}')
                    self.Debug(f"Trade amount: {trade_amount}")

                except Exception as e:
                    self.Debug(e)

    def prepare_data_trade(self):
        data = np.empty(shape=(len(self.current_instruments), 0, 4), dtype=np.float)

        i = 0

        for instrument in self.current_instruments:
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
