from clr import AddReference

AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Indicators")
AddReference("QuantConnect.Common")

from QuantConnect import *
from QuantConnect.Data import *
from QuantConnect.Algorithm import *
from QuantConnect.Brokerages import *

from datetime import timedelta

#   cd util_lib
#   python -m tensorboard.main --logdir=.
#   tensorboard --logdir=.

MARKET_NAME = "oanda"
RESOLUTION_NAME = "daily"
INSTRUMENTS = ['euraud', 'eurjpy', 'gbpaud', 'gbpjpy', 'nzdchf', 'nzdjpy', 'usdchf']
START_DATE = [2020, 1, 23]
END_DATE = [2020, 1, 24]
CASH = 100000.0
#   TRADE_LOOKBACK_PERIOD = 100
LEVERAGE = 20
WEIGHTS = [0.1352523, -0.13528316, 0.1263351, 0.13528033, -0.13395643, -0.12322688, 0.13525382, 0.07541198]


class main(QCAlgorithm):

    def Initialize(self):
        # instruments_attributes = load_objects_from_pickle(path_join(*[MODELS_DIR, MODEL_NAME, 'online', 'instruments.pkl']))
        # self.account_currency = instruments_attributes[0]
        # self.instruments = instruments_attributes[1]
        # self.pip_size = instruments_attributes[2]
        # self.pip_spread = instruments_attributes[3]

        if MARKET_NAME == 'oanda':
            self.market = Market.Oanda
            self.SetBrokerageModel(BrokerageName.OandaBrokerage)
        elif MARKET_NAME == 'fxcm':
            self.market = Market.FXCM
            self.SetBrokerageModel(BrokerageName.FxcmBrokerage)

        if RESOLUTION_NAME == 'daily':
            self.resolution = Resolution.Daily
        elif RESOLUTION_NAME == 'hour':
            self.resolution = Resolution.Hour
        elif RESOLUTION_NAME == 'minute':
            self.resolution = Resolution.Minute

        self.symbols = [self.AddForex(instrument, self.resolution, self.market).Symbol for instrument in INSTRUMENTS]

        #   self.SetStartDate(2021, 8, 14)
        self.SetStartDate(START_DATE[0], START_DATE[1], START_DATE[2])
        self.SetEndDate(END_DATE[0], END_DATE[1], END_DATE[2])

        self.SetCash(CASH)

        # self.rollingwindow = {}
        # symbols = self.Securities.Keys
        # history = self.History(symbols, TRADE_LOOKBACK_PERIOD, self.resolution)
        # for symbol in symbols:
        #     self.rollingwindow[symbol] = QuoteBarData(self, symbol, TRADE_LOOKBACK_PERIOD, history.loc[symbol],
        #                                               self.resolution)

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
        pass

    def rebalance(self):
        for i, instrument in enumerate(INSTRUMENTS):
            self.SetHoldings(instrument, LEVERAGE * WEIGHTS[i])

# class QuoteBarData:
#
#     def __init__(self, algorithm, symbol, lookback, history, resolution):
#         self.open = deque(maxlen=lookback)
#         self.high = deque(maxlen=lookback)
#         self.low = deque(maxlen=lookback)
#         self.close = deque(maxlen=lookback)
#         self.Debug = algorithm.Debug
#
#         self.time = algorithm.Time
#
#         if resolution == Resolution.Daily:
#             consolidator = QuoteBarConsolidator(timedelta(days=1))
#         elif resolution == Resolution.Hour:
#             consolidator = QuoteBarConsolidator(timedelta(hours=1))
#         elif resolution == Resolution.Minute:
#             consolidator = QuoteBarConsolidator(timedelta(minutes=1))
#
#         algorithm.SubscriptionManager.AddConsolidator(symbol, consolidator)
#         consolidator.DataConsolidated += self.OnDataConsolidated
#
#         for time, row in history.iterrows():
#             self.time = time
#             self.Update(time, row.open, row.high, row.low, row.close)
#
#     def Update(self, time, o, h, l, c):
#         self.time = time
#         self.open.append(o)
#         self.high.append(h)
#         self.low.append(l)
#         self.close.append(c)
#
#     def OnDataConsolidated(self, sender, bar):
#         self.Update(bar.EndTime, bar.Open, bar.High, bar.Low, bar.Close)
