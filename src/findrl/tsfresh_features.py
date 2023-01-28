import numpy as np
import pandas as pd
import talib


# from tsfresh import extract_features
# from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters
# from tsfresh.utilities.dataframe_functions import impute


# from .pf_stats import pf_stats
# from .stats import sharpe, smart_sharpe, rolling_sharpe, sortino, smart_sortino, rolling_sortino, adjusted_sortino, \
#     omega, gain_to_pain_ratio, cagr, rar, kurtosis, calmar, ulcer_index, ulcer_performance_index, serenity_index, \
#     risk_of_ruin, value_at_risk, conditional_value_at_risk, tail_ratio, payoff_ratio, profit_ratio, profit_factor, \
#     cpc_index, common_sense_ratio, outlier_win_ratio, outlier_loss_ratio, recovery_factor, risk_return_ratio, \
#     max_drawdown, kelly_criterion, information_ratio


class tsfresh_features:

    def __init__(self,
                 security,
                 open_name,
                 high_name,
                 low_name,
                 close_name):
        self.security = security
        self.open_name = open_name
        self.high_name = high_name
        self.low_name = low_name
        self.close_name = close_name

        self.open_price = self.security[self.open_name].values
        self.high_price = self.security[self.high_name].values
        self.low_price = self.security[self.low_name].values
        self.close_price = self.security[self.close_name].values

        #   print(self.security.info())
        #   print(self.security)

    def get_indicators_prices(self):
        #   Prices
        self.security['OPEN'] = self.open_price
        self.security['HIGH'] = self.high_price
        self.security['LOW'] = self.low_price
        self.security['CLOSE'] = self.close_price

        return self.security.dropna().astype(np.float)

    def get_indicators_returns(self):
        self.security['RR_1'] = self.security[self.close_name].fillna(1) / self.security[self.open_name].fillna(1)
        self.security['RR_2'] = self.security[self.high_name].fillna(1) / self.security[self.low_name].fillna(1)
        self.security['RR_3'] = self.security[self.high_name].fillna(1) / self.security[self.close_name].fillna(1)
        self.security['RR_4'] = self.security[self.close_name].fillna(1) / self.security[self.low_name].fillna(1)

        return self.security.dropna().astype(np.float)

    def get_indicators_log_returns(self):
        self.security['LOG_RR_1'] = np.log(self.security[self.close_name].fillna(1)) - np.log(
            self.security[self.open_name].fillna(1))
        self.security['LOG_RR_2'] = np.log(self.security[self.high_name].fillna(1)) - np.log(
            self.security[self.low_name].fillna(1))
        self.security['LOG_RR_3'] = np.log(self.security[self.high_name].fillna(1)) - np.log(
            self.security[self.close_name].fillna(1))
        self.security['LOG_RR_4'] = np.log(self.security[self.close_name].fillna(1)) - np.log(
            self.security[self.low_name].fillna(1))

        return self.security.dropna().astype(np.float)

    def get_indicators_patterns(self):
        self.security['CDL2CROWS'] = talib.CDL2CROWS(self.open_price, self.high_price, self.low_price, self.close_price)
        self.security['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(self.open_price, self.high_price, self.low_price,
                                                               self.close_price)
        self.security['CDL3INSIDE'] = talib.CDL3INSIDE(self.open_price, self.high_price, self.low_price,
                                                       self.close_price)
        self.security['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(self.open_price, self.high_price, self.low_price,
                                                               self.close_price)
        self.security['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(self.open_price, self.high_price, self.low_price,
                                                         self.close_price)
        self.security['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(self.open_price, self.high_price, self.low_price,
                                                                   self.close_price)
        self.security['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(self.open_price, self.high_price, self.low_price,
                                                                     self.close_price)
        self.security['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(self.open_price, self.high_price, self.low_price,
                                                                   self.close_price, penetration=0)
        self.security['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(self.open_price, self.high_price, self.low_price,
                                                                 self.close_price)
        self.security['CDLBELTHOLD'] = talib.CDLBELTHOLD(self.open_price, self.high_price, self.low_price,
                                                         self.close_price)
        self.security['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(self.open_price, self.high_price, self.low_price,
                                                           self.close_price)
        self.security['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(self.open_price, self.high_price, self.low_price,
                                                                       self.close_price)
        self.security['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(self.open_price, self.high_price,
                                                                         self.low_price, self.close_price)
        self.security['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(self.open_price, self.high_price, self.low_price,
                                                                   self.close_price)
        self.security['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(self.open_price, self.high_price, self.low_price,
                                                                     self.close_price, penetration=0)
        self.security['CDLDOJI'] = talib.CDLDOJI(self.open_price, self.high_price, self.low_price, self.close_price)
        self.security['CDLDOJISTAR'] = talib.CDLDOJISTAR(self.open_price, self.high_price, self.low_price,
                                                         self.close_price)
        self.security['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(self.open_price, self.high_price, self.low_price,
                                                                   self.close_price)
        self.security['CDLENGULFING'] = talib.CDLENGULFING(self.open_price, self.high_price, self.low_price,
                                                           self.close_price)
        self.security['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(self.open_price, self.high_price, self.low_price,
                                                                       self.close_price, penetration=0)
        self.security['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(self.open_price, self.high_price, self.low_price,
                                                               self.close_price, penetration=0)
        self.security['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(self.open_price, self.high_price,
                                                                         self.low_price, self.close_price)
        self.security['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(self.open_price, self.high_price, self.low_price,
                                                                     self.close_price)
        self.security['CDLHAMMER'] = talib.CDLHAMMER(self.open_price, self.high_price, self.low_price, self.close_price)
        self.security['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(self.open_price, self.high_price, self.low_price,
                                                             self.close_price)
        self.security['CDLHARAMI'] = talib.CDLHARAMI(self.open_price, self.high_price, self.low_price, self.close_price)
        self.security['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(self.open_price, self.high_price, self.low_price,
                                                               self.close_price)
        self.security['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(self.open_price, self.high_price, self.low_price,
                                                         self.close_price)
        self.security['CDLHIKKAKE'] = talib.CDLHIKKAKE(self.open_price, self.high_price, self.low_price,
                                                       self.close_price)
        self.security['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(self.open_price, self.high_price, self.low_price,
                                                             self.close_price)
        self.security['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(self.open_price, self.high_price, self.low_price,
                                                                 self.close_price)
        self.security['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(self.open_price, self.high_price, self.low_price,
                                                                       self.close_price)
        self.security['CDLINNECK'] = talib.CDLINNECK(self.open_price, self.high_price, self.low_price, self.close_price)
        self.security['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(self.open_price, self.high_price, self.low_price,
                                                                     self.close_price)
        self.security['CDLKICKING'] = talib.CDLKICKING(self.open_price, self.high_price, self.low_price,
                                                       self.close_price)
        self.security['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(self.open_price, self.high_price, self.low_price,
                                                                       self.close_price)
        self.security['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(self.open_price, self.high_price, self.low_price,
                                                                 self.close_price)
        self.security['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(self.open_price, self.high_price, self.low_price,
                                                                     self.close_price)
        self.security['CDLLONGLINE'] = talib.CDLLONGLINE(self.open_price, self.high_price, self.low_price,
                                                         self.close_price)
        self.security['CDLMARUBOZU'] = talib.CDLMARUBOZU(self.open_price, self.high_price, self.low_price,
                                                         self.close_price)
        self.security['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(self.open_price, self.high_price, self.low_price,
                                                               self.close_price)
        self.security['CDLMATHOLD'] = talib.CDLMATHOLD(self.open_price, self.high_price, self.low_price,
                                                       self.close_price, penetration=0)
        self.security['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(self.open_price, self.high_price, self.low_price,
                                                                       self.close_price, penetration=0)
        self.security['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(self.open_price, self.high_price, self.low_price,
                                                               self.close_price, penetration=0)
        self.security['CDLONNECK'] = talib.CDLONNECK(self.open_price, self.high_price, self.low_price, self.close_price)
        self.security['CDLPIERCING'] = talib.CDLPIERCING(self.open_price, self.high_price, self.low_price,
                                                         self.close_price)
        self.security['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(self.open_price, self.high_price, self.low_price,
                                                               self.close_price)
        self.security['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(self.open_price, self.high_price,
                                                                         self.low_price, self.close_price)
        self.security['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(self.open_price, self.high_price, self.low_price,
                                                                       self.close_price)
        self.security['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(self.open_price, self.high_price, self.low_price,
                                                                 self.close_price)
        self.security['CDLSHORTLINE'] = talib.CDLSHORTLINE(self.open_price, self.high_price, self.low_price,
                                                           self.close_price)
        self.security['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(self.open_price, self.high_price, self.low_price,
                                                               self.close_price)
        self.security['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(self.open_price, self.high_price, self.low_price,
                                                                     self.close_price)
        self.security['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(self.open_price, self.high_price, self.low_price,
                                                                   self.close_price)
        self.security['CDLTAKURI'] = talib.CDLTAKURI(self.open_price, self.high_price, self.low_price, self.close_price)
        self.security['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(self.open_price, self.high_price, self.low_price,
                                                           self.close_price)
        self.security['CDLTHRUSTING'] = talib.CDLTHRUSTING(self.open_price, self.high_price, self.low_price,
                                                           self.close_price)
        self.security['CDLTRISTAR'] = talib.CDLTRISTAR(self.open_price, self.high_price, self.low_price,
                                                       self.close_price)
        self.security['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(self.open_price, self.high_price, self.low_price,
                                                                 self.close_price)
        self.security['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(self.open_price, self.high_price, self.low_price,
                                                                       self.close_price)
        self.security['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(self.open_price, self.high_price,
                                                                         self.low_price, self.close_price)

        return self.security.dropna().astype(np.float)

    def get_indicators_overlap_studies(self):
        self.security['UPPERBAND_CLOSE'], self.security['MIDDLEBAND_CLOSE'], self.security[
            'LOWERBAND_CLOSE'] = talib.BBANDS(
            self.close_price)
        self.security['UPPERBAND_HIGH'], self.security['MIDDLEBAND_HIGH'], self.security[
            'LOWERBAND_HIGH'] = talib.BBANDS(
            self.high_price)
        self.security['UPPERBAND_LOW'], self.security['MIDDLEBAND_LOW'], self.security['LOWERBAND_LOW'] = talib.BBANDS(
            self.low_price)

        self.security['DEMA_CLOSE'] = talib.DEMA(self.close_price)
        self.security['DEMA_HIGH'] = talib.DEMA(self.high_price)
        self.security['DEMA_LOW'] = talib.DEMA(self.low_price)

        self.security['EMA_CLOSE'] = talib.EMA(self.close_price)
        self.security['EMA_HIGH'] = talib.EMA(self.high_price)
        self.security['EMA_LOW'] = talib.EMA(self.low_price)

        self.security['HT_TRENDLINE_CLOSE'] = talib.HT_TRENDLINE(self.close_price)
        self.security['HT_TRENDLINE_HIGH'] = talib.HT_TRENDLINE(self.high_price)
        self.security['HT_TRENDLINE_LOW'] = talib.HT_TRENDLINE(self.low_price)

        self.security['KAMA_CLOSE'] = talib.KAMA(self.close_price)
        self.security['KAMA_HIGH'] = talib.KAMA(self.high_price)
        self.security['KAMA_LOW'] = talib.KAMA(self.low_price)

        self.security['MA_CLOSE'] = talib.MA(self.close_price)
        self.security['MA_HIGH'] = talib.MA(self.high_price)
        self.security['MA_LOW'] = talib.MA(self.low_price)

        self.security['MAMA_CLOSE'], self.security['FAMA_CLOSE'] = talib.MAMA(self.close_price)
        self.security['MAMA_HIGH'], self.security['FAMA_HIGH'] = talib.MAMA(self.high_price)
        self.security['MAMA_LOW'], self.security['FAMA_LOW'] = talib.MAMA(self.low_price)

        #   self.security['MAVP'] = talib.MAVP(self.close_price, periods)
        self.security['MIDPOINT_CLOSE'] = talib.MIDPOINT(self.close_price)
        self.security['MIDPOINT_HIGH'] = talib.MIDPOINT(self.high_price)
        self.security['MIDPOINT_LOW'] = talib.MIDPOINT(self.low_price)

        self.security['MIDPRICE'] = talib.MIDPRICE(self.high_price, self.low_price)
        self.security['SAR'] = talib.SAR(self.high_price, self.low_price)
        self.security['SAREXT'] = talib.SAREXT(self.high_price, self.low_price)

        self.security['SMA_CLOSE'] = talib.SMA(self.close_price)
        self.security['SMA_HIGH'] = talib.SMA(self.high_price)
        self.security['SMA_LOW'] = talib.SMA(self.low_price)

        self.security['T3_CLOSE'] = talib.T3(self.close_price)
        self.security['T3_HIGH'] = talib.T3(self.high_price)
        self.security['T3_LOW'] = talib.T3(self.low_price)

        self.security['TEMA_CLOSE'] = talib.TEMA(self.close_price)
        self.security['TEMA_HIGH'] = talib.TEMA(self.high_price)
        self.security['TEMA_LOW'] = talib.TEMA(self.low_price)

        self.security['TRIMA_CLOSE'] = talib.TRIMA(self.close_price)
        self.security['TRIMA_HIGH'] = talib.TRIMA(self.high_price)
        self.security['TRIMA_LOW'] = talib.TRIMA(self.low_price)

        self.security['WMA_CLOSE'] = talib.WMA(self.close_price)
        self.security['WMA_HIGH'] = talib.WMA(self.high_price)
        self.security['WMA_LOW'] = talib.WMA(self.low_price)

        return self.security.dropna().astype(np.float)

    def get_indicators_momentum(self):
        self.security['ADX'] = talib.ADX(self.high_price, self.low_price, self.close_price)
        self.security['ADXR'] = talib.ADXR(self.high_price, self.low_price, self.close_price)

        self.security['APO_CLOSE'] = talib.APO(self.close_price)
        self.security['APO_HIGH'] = talib.APO(self.high_price)
        self.security['APO_LOW'] = talib.APO(self.low_price)

        self.security['AROONDOWN'], self.security['AROONUP'] = talib.AROON(self.high_price, self.low_price)
        self.security['AROONOSC'] = talib.AROONOSC(self.high_price, self.low_price)
        self.security['BOP'] = talib.BOP(self.open_price, self.high_price, self.low_price, self.close_price)
        self.security['CCI'] = talib.CCI(self.high_price, self.low_price, self.close_price)

        self.security['CMO_CLOSE'] = talib.CMO(self.close_price)
        self.security['CMO_HIGH'] = talib.CMO(self.high_price)
        self.security['CMO_LOW'] = talib.CMO(self.low_price)

        self.security['DX'] = talib.DX(self.high_price, self.low_price, self.close_price)

        self.security['MACD_CLOSE'], self.security['MACDSIGNAL_CLOSE'], self.security['MACDHIST_CLOSE'] = talib.MACD(
            self.close_price)
        self.security['MACD_HIGH'], self.security['MACDSIGNAL_HIGH'], self.security['MACDHIST_HIGH'] = talib.MACD(
            self.high_price)
        self.security['MACD_LOW'], self.security['MACDSIGNAL_LOW'], self.security['MACDHIST_LOW'] = talib.MACD(
            self.low_price)

        self.security['MACDEXT_CLOSE'], self.security['MACDEXTSIGNAL_CLOSE'], self.security[
            'MACDEXTHIST_CLOSE'] = talib.MACDEXT(
            self.close_price)
        self.security['MACDEXT_HIGH'], self.security['MACDEXTSIGNAL_HIGH'], self.security[
            'MACDEXTHIST_HIGH'] = talib.MACDEXT(
            self.high_price)
        self.security['MACDEXT_LOW'], self.security['MACDEXTSIGNAL_LOW'], self.security[
            'MACDEXTHIST_LOW'] = talib.MACDEXT(
            self.low_price)

        self.security['MACDFIX_CLOSE'], self.security['MACDSIGNALFIX_CLOSE'], self.security[
            'MACDHISTFIX_CLOSE'] = talib.MACDFIX(
            self.close_price)
        self.security['MACDFIX_HIGH'], self.security['MACDSIGNALFIX_HIGH'], self.security[
            'MACDHISTFIX_HIGH'] = talib.MACDFIX(
            self.high_price)
        self.security['MACDFIX_LOW'], self.security['MACDSIGNALFIX_LOW'], self.security[
            'MACDHISTFIX_LOW'] = talib.MACDFIX(
            self.low_price)

        self.security['MINUS_DI'] = talib.MINUS_DI(self.high_price, self.low_price, self.close_price)
        self.security['MINUS_DM'] = talib.MINUS_DM(self.high_price, self.low_price)

        self.security['MOM_CLOSE'] = talib.MOM(self.close_price)
        self.security['MOM_HIGH'] = talib.MOM(self.high_price)
        self.security['MOM_LOW'] = talib.MOM(self.low_price)

        self.security['PLUS_DI'] = talib.PLUS_DI(self.high_price, self.low_price, self.close_price)
        self.security['PLUS_DM'] = talib.PLUS_DM(self.high_price, self.low_price)

        self.security['PPO_CLOSE'] = talib.PPO(self.close_price)
        self.security['PPO_HIGH'] = talib.PPO(self.high_price)
        self.security['PPO_LOW'] = talib.PPO(self.low_price)

        self.security['ROC_CLOSE'] = talib.ROC(self.close_price)
        self.security['ROC_HIGH'] = talib.ROC(self.high_price)
        self.security['ROC_LOW'] = talib.ROC(self.low_price)

        self.security['ROCP_CLOSE'] = talib.ROCP(self.close_price)
        self.security['ROCP_HIGH'] = talib.ROCP(self.high_price)
        self.security['ROCP_LOW'] = talib.ROCP(self.low_price)

        self.security['ROCR_CLOSE'] = talib.ROCR(self.close_price)
        self.security['ROCR_HIGH'] = talib.ROCR(self.high_price)
        self.security['ROCR_LOW'] = talib.ROCR(self.low_price)

        self.security['ROCR100_CLOSE'] = talib.ROCR100(self.close_price)
        self.security['ROCR100_HIGH'] = talib.ROCR100(self.high_price)
        self.security['ROCR100_LOW'] = talib.ROCR100(self.low_price)

        self.security['RSI_CLOSE'] = talib.RSI(self.close_price)
        self.security['RSI_HIGH'] = talib.RSI(self.high_price)
        self.security['RSI_LOW'] = talib.RSI(self.low_price)

        self.security['STOCH_SLOWK'], self.security['STOCH_SLOWD'] = talib.STOCH(self.high_price, self.low_price,
                                                                                 self.close_price)
        self.security['STOCHF_FASTK'], self.security['STOCHF_FASTD'] = talib.STOCHF(self.high_price, self.low_price,
                                                                                    self.close_price)

        self.security['STOCHRSI_FASTK_CLOSE'], self.security['STOCHRSI_FASTD_CLOSE'] = talib.STOCHRSI(self.close_price)
        self.security['STOCHRSI_FASTK_HIGH'], self.security['STOCHRSI_FASTD_HIGH'] = talib.STOCHRSI(self.high_price)
        self.security['STOCHRSI_FASTK_LOW'], self.security['STOCHRSI_FASTD_LOW'] = talib.STOCHRSI(self.low_price)

        self.security['TRIX_CLOSE'] = talib.TRIX(self.close_price)
        self.security['TRIX_HIGH'] = talib.TRIX(self.high_price)
        self.security['TRIX_LOW'] = talib.TRIX(self.low_price)

        self.security['ULTOSC'] = talib.ULTOSC(self.high_price, self.low_price, self.close_price)
        self.security['WILLR'] = talib.WILLR(self.high_price, self.low_price, self.close_price)

        return self.security.dropna().astype(np.float)

    def get_indicators_volatility(self):
        self.security['ATR'] = talib.ATR(self.high_price, self.low_price, self.close_price)
        self.security['NATR'] = talib.NATR(self.high_price, self.low_price, self.close_price)
        self.security['TRANGE'] = talib.TRANGE(self.high_price, self.low_price, self.close_price)

        return self.security.dropna().astype(np.float)

    def get_indicators_price_transform(self):
        self.security['AVGPRICE'] = talib.AVGPRICE(self.open_price, self.high_price, self.low_price, self.close_price)
        self.security['MEDPRICE'] = talib.MEDPRICE(self.high_price, self.low_price)
        self.security['TYPPRICE'] = talib.TYPPRICE(self.high_price, self.low_price, self.close_price)
        self.security['WCLPRICE'] = talib.WCLPRICE(self.high_price, self.low_price, self.close_price)

        return self.security.dropna().astype(np.float)

    def get_indicators_cycle_indicators(self):
        self.security['HT_DCPERIOD_CLOSE'] = talib.HT_DCPERIOD(self.close_price)
        self.security['HT_DCPERIOD_HIGH'] = talib.HT_DCPERIOD(self.high_price)
        self.security['HT_DCPERIOD_LOW'] = talib.HT_DCPERIOD(self.low_price)

        self.security['HT_DCPHASE_CLOSE'] = talib.HT_DCPHASE(self.close_price)
        self.security['HT_DCPHASE_HIGH'] = talib.HT_DCPHASE(self.high_price)
        self.security['HT_DCPHASE_LOW'] = talib.HT_DCPHASE(self.low_price)

        self.security['INPHASE_CLOSE'], self.security['QUADRATURE_CLOSE'] = talib.HT_PHASOR(self.close_price)
        self.security['INPHASE_HIGH'], self.security['QUADRATURE_HIGH'] = talib.HT_PHASOR(self.high_price)
        self.security['INPHASE_LOW'], self.security['QUADRATURE_LOW'] = talib.HT_PHASOR(self.low_price)

        self.security['SINE_CLOSE'], self.security['LEADSINE_CLOSE'] = talib.HT_SINE(self.close_price)
        self.security['SINE_HIGH'], self.security['LEADSINE_HIGH'] = talib.HT_SINE(self.high_price)
        self.security['SINE_LOW'], self.security['LEADSINE_LOW'] = talib.HT_SINE(self.low_price)

        self.security['HT_TRENDMODE_CLOSE'] = talib.HT_TRENDMODE(self.close_price)
        self.security['HT_TRENDMODE_HIGH'] = talib.HT_TRENDMODE(self.high_price)
        self.security['HT_TRENDMODE_LOW'] = talib.HT_TRENDMODE(self.low_price)

        return self.security.dropna().astype(np.float)

    def get_indicators_statistic(self):
        self.security['BETA'] = talib.BETA(self.high_price, self.low_price)
        self.security['CORREL'] = talib.CORREL(self.high_price, self.low_price)

        self.security['LINEARREG_CLOSE'] = talib.LINEARREG(self.close_price)
        self.security['LINEARREG_HIGH'] = talib.LINEARREG(self.high_price)
        self.security['LINEARREG_LOW'] = talib.LINEARREG(self.low_price)

        self.security['LINEARREG_ANGLE_CLOSE'] = talib.LINEARREG_ANGLE(self.close_price)
        self.security['LINEARREG_ANGLE_HIGH'] = talib.LINEARREG_ANGLE(self.high_price)
        self.security['LINEARREG_ANGLE_LOW'] = talib.LINEARREG_ANGLE(self.low_price)

        self.security['LINEARREG_INTERCEPT_CLOSE'] = talib.LINEARREG_INTERCEPT(self.close_price)
        self.security['LINEARREG_INTERCEPT_HIGH'] = talib.LINEARREG_INTERCEPT(self.high_price)
        self.security['LINEARREG_INTERCEPT_LOW'] = talib.LINEARREG_INTERCEPT(self.low_price)

        self.security['LINEARREG_SLOPE_CLOSE'] = talib.LINEARREG_SLOPE(self.close_price)
        self.security['LINEARREG_SLOPE_HIGH'] = talib.LINEARREG_SLOPE(self.high_price)
        self.security['LINEARREG_SLOPE_LOW'] = talib.LINEARREG_SLOPE(self.low_price)

        self.security['STDDEV_CLOSE'] = talib.STDDEV(self.close_price)
        self.security['STDDEV_HIGH'] = talib.STDDEV(self.high_price)
        self.security['STDDEV_LOW'] = talib.STDDEV(self.low_price)

        self.security['TSF_CLOSE'] = talib.TSF(self.close_price)
        self.security['TSF_HIGH'] = talib.TSF(self.high_price)
        self.security['TSF_LOW'] = talib.TSF(self.low_price)

        self.security['VAR_CLOSE'] = talib.VAR(self.close_price)
        self.security['VAR_HIGH'] = talib.VAR(self.high_price)
        self.security['VAR_LOW'] = talib.VAR(self.low_price)

        return self.security.dropna().astype(np.float)

    def get_indicators_math_transform(self):
        self.security['ACOS'] = talib.ACOS(self.close_price)
        self.security['ASIN'] = talib.ASIN(self.close_price)
        self.security['ATAN'] = talib.ATAN(self.close_price)
        self.security['CEIL'] = talib.CEIL(self.close_price)
        self.security['COS'] = talib.COS(self.close_price)
        self.security['COSH'] = talib.COSH(self.close_price)
        self.security['EXP'] = talib.EXP(self.close_price)
        self.security['FLOOR'] = talib.FLOOR(self.close_price)
        self.security['LN'] = talib.LN(self.close_price)
        self.security['LOG10'] = talib.LOG10(self.close_price)
        self.security['SIN'] = talib.SIN(self.close_price)
        self.security['SINH'] = talib.SINH(self.close_price)
        self.security['SQRT'] = talib.SQRT(self.close_price)
        self.security['TAN'] = talib.TAN(self.close_price)
        self.security['TANH'] = talib.TANH(self.close_price)

        return self.security.dropna().astype(np.float)

    def get_indicators_math_operator(self):
        self.security['ADD'] = talib.ADD(self.high_price, self.low_price)
        self.security['DIV'] = talib.DIV(self.high_price, self.low_price)

        self.security['MAX_CLOSE'] = talib.MAX(self.close_price)
        self.security['MAX_HIGH'] = talib.MAX(self.high_price)
        self.security['MAX_LOW'] = talib.MAX(self.low_price)

        self.security['MIN_CLOSE'] = talib.MIN(self.close_price)
        self.security['MIN_HIGH'] = talib.MIN(self.high_price)
        self.security['MIN_LOW'] = talib.MIN(self.low_price)

        self.security['MULT'] = talib.MULT(self.high_price, self.low_price)
        self.security['SUB'] = talib.SUB(self.high_price, self.low_price)

        self.security['SUM_CLOSE'] = talib.SUM(self.close_price)
        self.security['SUM_HIGH'] = talib.SUM(self.high_price)
        self.security['SUM_LOW'] = talib.SUM(self.low_price)

        return self.security.dropna().astype(np.float)

    def get_indicators_all(self, method):
        if method == 'minimal':
            self.get_features_minimal()
        elif method == 'efficient':
            self.get_features_efficient()
        elif method == 'comprehensive':
            self.get_features_comprehensive()

        #   Prices
        #   self.get_indicators_prices()

        #   Returns
        self.get_indicators_returns()

        #   Log Returns
        self.get_indicators_log_returns()

        #   Pattern Recognition
        self.get_indicators_patterns()

        #   Overlap Studies
        self.get_indicators_overlap_studies()

        #   Momentum Indicators
        self.get_indicators_momentum()

        #   Volatility Indicators
        self.get_indicators_volatility()

        #   Price Transform
        self.get_indicators_price_transform()

        #   Cycle Indicators
        self.get_indicators_cycle_indicators()

        #   Statistic
        self.get_indicators_statistic()

        #   Math Transform
        # self.get_indicators_math_transform()

        #   Math Operator
        self.get_indicators_math_operator()

        # self.get_indicators_performance(self.security['RR'], 100)
        #
        # self.get_indicators_metrics(self.security['RR'])

        # self.get_indicators_hurst()

        #   self.security.drop([self.open_name, self.close_name, self.high_name, self.low_name], axis=1)

        #   self.security.reset_index(inplace=True)

        self.security = self.security.dropna().astype(np.float)

        print(self.security.info())
        print(self.security.columns)
        print(self.security)

        return self.security

    def get_features_minimal(self):
        data = self.security

        data = self.security

        data = data.reset_index(drop=False)

        extraction_settings = MinimalFCParameters()

        # Extract features
        # extracted_features = extract_features(data, column_id="index", column_sort="index",
        #                                       default_fc_parameters=extraction_settings, impute_function=impute)

        extracted_features = extract_features(data, column_id="index", column_sort="index",
                                              default_fc_parameters=extraction_settings)

        # Drop features with NaN
        extracted_features_clean = extracted_features.dropna(axis=1, how='all').reset_index(drop=True)

        # Drop features with constants
        extracted_features_clean = extracted_features_clean.loc[:,
                                   (extracted_features_clean != extracted_features_clean.iloc[0]).any()]

        #   self.security = pd.concat([self.security, extracted_features_clean.dropna().astype(np.float)], axis=1)

        self.security = pd.concat([self.security, extracted_features], axis=1)

        return self.security.dropna().astype(np.float)

    def get_features_efficient(self):
        data = self.security

        data = data.reset_index(drop=False)

        extraction_settings = EfficientFCParameters()

        # Extract features
        extracted_features = extract_features(data, column_id="index", column_sort="index",
                                              default_fc_parameters=extraction_settings, impute_function=impute)

        # Drop features with NaN
        extracted_features_clean = extracted_features.dropna(axis=1, how='all').reset_index(drop=True)

        # Drop features with constants
        extracted_features_clean = extracted_features_clean.loc[:,
                                   (extracted_features_clean != extracted_features_clean.iloc[0]).any()]

        self.security = pd.concat([self.security, extracted_features_clean.dropna().astype(np.float)], axis=1)

        return self.security.dropna().astype(np.float)

    def get_features_comprehensive(self):
        data = self.security

        data = data.reset_index(drop=False)

        extraction_settings = ComprehensiveFCParameters()

        # Extract features
        extracted_features = extract_features(data, column_id="index", column_sort="index",
                                              default_fc_parameters=extraction_settings, impute_function=impute)

        # Drop features with NaN
        extracted_features_clean = extracted_features.dropna(axis=1, how='all').reset_index(drop=True)

        # Drop features with constants
        extracted_features_clean = extracted_features_clean.loc[:,
                                   (extracted_features_clean != extracted_features_clean.iloc[0]).any()]

        self.security = pd.concat([self.security, extracted_features_clean.dropna().astype(np.float)], axis=1)

        return self.security.dropna().astype(np.float)
