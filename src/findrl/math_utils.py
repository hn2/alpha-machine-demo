import numpy as np


class math_utils:
    EPS = np.finfo(float).eps
    GAMMA = 0.5
    NEGATIVE_REWARD = -1

    def my_round(x):
        return np.round(x, 4)

    def my_sum_abs(x):
        ''' Sum of absolute values '''
        return np.sum(np.abs(x))

    def my_add(x, y):
        result = []
        i = 0
        for v in (y):
            if x[i] > 0:
                result.append(x[i] + v)
                #   result.append((x[i] + v) if (x[i] + v) > 0 else 0)
            elif x[i] < 0:
                result.append(x[i] - v)
                #   result.append((x[i] - v) if (x[i] - v) < 0 else 0)
            else:
                result.append(x[i])
            i += 1

        return result

    def my_subtract(x, y):
        result = []
        i = 0
        for v in (y):
            if x[i] > 0:
                result.append(x[i] - v)
            elif x[i] < 0:
                result.append(x[i] + v)
            else:
                result.append(x[i])
            i += 1

        return result

    def my_add_weights(weights_long, weights_short):
        result = []
        i = 0
        for v in (weights_long):
            if weights_long[i] > 0 and weights_short[i] == 0:
                result.append(weights_long[i])
            elif weights_long[i] == 0 and weights_short[i] < 0:
                result.append(weights_short[i])
            else:
                result.append(0)
            i += 1

        return result

    def my_multiply_with_minimum_fees(commission_fee, minimum_fee, trade_amounts):
        result = []
        for v in (trade_amounts):
            if v * commission_fee < minimum_fee:
                result.append(v * commission_fee)
            else:
                result.append(minimum_fee)

        return result

    def almost(a, b, decimal=6, fill_value=False):
        """
        Returns True if a and b are equal up to decimal places.

        If fill_value is True, masked values considered equal. Otherwise,
        masked values are considered unequal.

        """
        m = np.ma.mask_or(np.ma.getmask(a), np.ma.getmask(b))
        d1 = np.ma.filled(a)
        d2 = np.ma.filled(b)
        if d1.dtype.char == "O" or d2.dtype.char == "O":
            return np.equal(d1, d2).ravel()
        x = np.ma.filled(np.ma.masked_array(d1, copy=False, mask=m), fill_value).astype(float)
        y = np.ma.filled(np.ma.masked_array(d2, copy=False, mask=m), 1).astype(float)
        d = np.around(np.abs(x - y), decimal) <= 10.0 ** (-decimal)
        return d.ravel()

    def get_normelize_weights(weights):
        weights /= (math_utils.my_sum_abs(weights) + math_utils.EPS)
        weights[-1] += np.clip(1 - math_utils.my_sum_abs(weights), 0,
                               1)  # if weights are all zeros we normalise to [0,0..1]
        weights[-1] = math_utils.my_sum_abs(weights[-1])

        if np.all(np.isnan(weights)):
            weights[:] = 0
            weights[-1] = 1

        #   np.testing.assert_almost_equal(my_math.my_sum_abs(weights), 1.0, 3, 'absolute weights should sum to 1. weights=%s' %weights)

        assert ((weights[0:-1] >= -1) * (weights[0:-1] <= 1) * (weights[-1] >= 0) * (
                weights[-1] <= 1)).all(), 'all weights values should be between -1 and 1. Not %s' % weights

        return weights

    def get_hurst_exponent(time_series, max_lag=32):
        """Returns the Hurst Exponent of the time series"""

        lags = range(2, max_lag)

        # variances of the lagged differences
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

        # calculate the slope of the log plot -> the Hurst Exponent
        reg = np.polyfit(np.log(lags), np.log(tau), 1)

        return reg[0]

    # def get_acf(time_series, lag=32):
    #     """Returns the Hurst Exponent of the time series"""
    #
    #     mu = np.mean(time_series)
    #     acf = np.dot((time_series - mu).iloc[lag:], (time_series.shift(lag) - mu).iloc[lag:]) / sum((time_series - mu) ** 2)
    #
    #     return reg[0]
