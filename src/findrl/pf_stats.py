import math

import numpy
import numpy.random as nrand

"""
http://www.turingfinance.com/computational-investing-with-python-week-one/
Note - for some of the metrics the absolute value is returns. This is because if the risk (loss) is higher we want to
discount the expected excess return from the portfolio by a higher amount. Therefore risk should be positive.
"""


class pf_stats:

    def __init__(self):

        super(pf_stats, self).__init__()

    def vol(self, returns):
        # Return the standard deviation of returns
        return numpy.std(returns)

    def beta(self, returns, market):
        # Create a matrix of [returns, market]
        m = numpy.matrix([returns, market])
        # Return the covariance of m divided by the standard deviation of the market returns
        return numpy.cov(m)[0][1] / numpy.std(market)

    def lpm(self, returns, threshold, order):
        # This method returns a lower partial moment of the returns
        # Create an array he same length as returns containing the minimum return threshold
        threshold_array = numpy.empty(len(returns))
        threshold_array.fill(threshold)
        # Calculate the difference between the threshold and the returns
        diff = threshold_array - returns
        # Set the minimum of each to 0
        diff = diff.clip(min=0)
        # Return the sum of the different to the power of order
        return numpy.sum(diff ** order) / len(returns)

    def hpm(self, returns, threshold, order):
        # This method returns a higher partial moment of the returns
        # Create an array he same length as returns containing the minimum return threshold
        threshold_array = numpy.empty(len(returns))
        threshold_array.fill(threshold)
        # Calculate the difference between the returns and the threshold
        diff = returns - threshold_array
        # Set the minimum of each to 0
        diff = diff.clip(min=0)
        # Return the sum of the different to the power of order
        return numpy.sum(diff ** order) / len(returns)

    def var(self, returns, alpha):
        # This method calculates the historical simulation var of the returns
        sorted_returns = numpy.sort(returns)
        # Calculate the index associated with alpha
        index = int(alpha * len(sorted_returns))
        # VaR should be positive
        return abs(sorted_returns[index])

    def cvar(self, returns, alpha):
        # This method calculates the condition VaR of the returns
        sorted_returns = numpy.sort(returns)
        # Calculate the index associated with alpha
        index = int(alpha * len(sorted_returns))
        # Calculate the total VaR beyond alpha
        sum_var = sorted_returns[0]
        for i in range(1, index):
            sum_var += sorted_returns[i]
        # Return the average VaR
        # CVaR should be positive
        return abs(sum_var / index)

    def prices(self, returns, base):
        # Converts returns into prices
        s = [base]
        for i in range(len(returns)):
            s.append(base * (1 + returns[i]))
        return numpy.array(s)

    def dd(self, returns, tau):
        # Returns the draw-down given time period tau
        values = self.prices(returns, 100)
        pos = len(values) - 1
        pre = pos - tau
        drawdown = float('+inf')
        # Find the maximum drawdown given tau
        while pre >= 0:
            dd_i = (values[pos] / values[pre]) - 1
            if dd_i < drawdown:
                drawdown = dd_i
            pos, pre = pos - 1, pre - 1
        # Drawdown should be positive
        return abs(drawdown)

    def max_dd(self, returns):
        # Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
        max_drawdown = float('-inf')
        for i in range(0, len(returns)):
            drawdown_i = self.dd(returns, i)
            if drawdown_i > max_drawdown:
                max_drawdown = drawdown_i
        # Max draw-down should be positive
        return abs(max_drawdown)

    def average_dd(self, returns, periods):
        # Returns the average maximum drawdown over n periods
        drawdowns = []
        for i in range(0, len(returns)):
            drawdown_i = self.dd(returns, i)
            drawdowns.append(drawdown_i)
        drawdowns = sorted(drawdowns)
        total_dd = abs(drawdowns[0])
        for i in range(1, periods):
            total_dd += abs(drawdowns[i])
        return total_dd / periods

    def average_dd_squared(self, returns, periods):
        # Returns the average maximum drawdown squared over n periods
        drawdowns = []
        for i in range(0, len(returns)):
            drawdown_i = math.pow(self.dd(returns, i), 2.0)
            drawdowns.append(drawdown_i)
        drawdowns = sorted(drawdowns)
        total_dd = abs(drawdowns[0])
        for i in range(1, periods):
            total_dd += abs(drawdowns[i])
        return total_dd / periods

    def treynor_ratio(self, er, returns, market, rf):
        return (er - rf) / self.beta(returns, market)

    def sharpe_ratio(self, er, returns, rf):
        return (er - rf) / self.vol(returns)

    def information_ratio(self, returns, benchmark):
        diff = returns - benchmark
        return numpy.mean(diff) / self.vol(diff)

    def modigliani_ratio(self, er, returns, benchmark, rf):
        np_rf = numpy.empty(len(returns))
        np_rf.fill(rf)
        rdiff = returns - np_rf
        bdiff = benchmark - np_rf
        return (er - rf) * (self.vol(rdiff) / self.vol(bdiff)) + rf

    def excess_var(self, er, returns, rf, alpha):
        return (er - rf) / self.var(returns, alpha)

    def conditional_sharpe_ratio(self, er, returns, rf, alpha):
        return (er - rf) / self.cvar(returns, alpha)

    def omega_ratio(self, er, returns, rf, target=0):
        return (er - rf) / self.lpm(returns, target, 1)

    def sortino_ratio(self, er, returns, rf, target=0):
        return (er - rf) / math.sqrt(self.lpm(returns, target, 2))

    def kappa_three_ratio(self, er, returns, rf, target=0):
        return (er - rf) / math.pow(self.lpm(returns, target, 3), float(1 / 3))

    def gain_loss_ratio(self, returns, target=0):
        return self.hpm(returns, target, 1) / self.lpm(returns, target, 1)

    def upside_potential_ratio(self, returns, target=0):
        return self.hpm(returns, target, 1) / math.sqrt(self.lpm(returns, target, 2))

    def calmar_ratio(self, er, returns, rf):
        return (er - rf) / self.max_dd(returns)

    def sterling_ratio(self, er, returns, rf, periods):
        return (er - rf) / self.average_dd(returns, periods)

    def burke_ratio(self, er, returns, rf, periods):
        return (er - rf) / math.sqrt(self.average_dd_squared(returns, periods))

    def test_risk_metrics(self):
        # This is just a testing method
        r = nrand.uniform(-1, 1, 50)
        m = nrand.uniform(-1, 1, 50)
        print("vol =", self.vol(r))
        print("beta =", self.beta(r, m))
        print("hpm(0.0)_1 =", self.hpm(r, 0.0, 1))
        print("lpm(0.0)_1 =", self.lpm(r, 0.0, 1))
        print("VaR(0.05) =", self.var(r, 0.05))
        print("CVaR(0.05) =", self.cvar(r, 0.05))
        print("Drawdown(5) =", self.dd(r, 5))
        print("Max Drawdown =", self.max_dd(r))

    def test_risk_adjusted_metrics(self):
        # Returns from the portfolio (r) and market (m)
        r = nrand.uniform(-1, 1, 50)
        m = nrand.uniform(-1, 1, 50)
        # Expected return
        e = numpy.mean(r)
        # Risk free rate
        f = 0.06
        # Risk-adjusted return based on Volatility
        print("Treynor Ratio =", self.treynor_ratio(e, r, m, f))
        print("Sharpe Ratio =", self.sharpe_ratio(e, r, f))
        print("Information Ratio =", self.information_ratio(r, m))
        # Risk-adjusted return based on Value at Risk
        print("Excess VaR =", self.excess_var(e, r, f, 0.05))
        print("Conditional Sharpe Ratio =", self.conditional_sharpe_ratio(e, r, f, 0.05))
        # Risk-adjusted return based on Lower Partial Moments
        print("Omega Ratio =", self.omega_ratio(e, r, f))
        print("Sortino Ratio =", self.sortino_ratio(e, r, f))
        print("Kappa 3 Ratio =", self.kappa_three_ratio(e, r, f))
        print("Gain Loss Ratio =", self.gain_loss_ratio(r))
        print("Upside Potential Ratio =", self.upside_potential_ratio(r))
        # Risk-adjusted return based on Drawdown risk
        print("Calmar Ratio =", self.calmar_ratio(e, r, f))
        print("Sterling Ratio =", self.sterling_ratio(e, r, f, 5))
        print("Burke Ratio =", self.burke_ratio(e, r, f, 5))
