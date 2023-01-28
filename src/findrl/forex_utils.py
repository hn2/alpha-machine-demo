#   https://fxssi.com/the-most-traded-currency-pairs
import random

from .convert import convert


def get_forex_1(spread):
    instruments = ['audusd']
    pip_size = [0.0001]
    pip_spread = [spread] * 1

    return instruments, pip_size, pip_spread


def get_forex_4(spread):
    instruments = ['eurusd', 'gbpusd', 'audusd', 'nzdusd']
    pip_size = [0.0001, 0.0001, 0.0001, 0.0001]
    pip_spread = [spread] * 4

    return instruments, pip_size, pip_spread


def get_forex_6(spread):
    instruments = ['eurusd', 'usdjpy', 'gbpusd', 'audusd', 'usdcad', 'usdchf']
    pip_size = [0.0001, 0.01, 0.0001, 0.0001, 0.0001, 0.0001]
    pip_spread = [spread] * 6

    return instruments, pip_size, pip_spread


def get_forex_7(spread):
    instruments = ['eurusd', 'usdjpy', 'gbpusd', 'audusd', 'usdcad', 'usdchf', 'nzdusd']
    pip_size = [0.0001, 0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
    pip_spread = [spread] * 7

    return instruments, pip_size, pip_spread


def get_forex_10(spread):
    instruments = ['eurusd', 'usdjpy', 'gbpusd', 'audusd', 'usdcad', 'usdchf', 'nzdusd',
                   'eurjpy', 'gbpjpy', 'eurgbp']
    pip_size = [0.0001, 0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                0.01, 0.01, 0.0001]
    pip_spread = [spread] * 12

    return instruments, pip_size, pip_spread


def get_forex_12(spread):
    instruments = ['eurusd', 'usdjpy', 'gbpusd', 'audusd', 'usdcad', 'usdchf', 'nzdusd',
                   'eurjpy', 'gbpjpy', 'eurgbp', 'audjpy', 'euraud']
    pip_size = [0.0001, 0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                0.01, 0.01, 0.0001, 0.01, 0.0001]
    pip_spread = [spread] * 12

    return instruments, pip_size, pip_spread


def get_forex_14(spread):
    instruments = ['eurusd', 'usdjpy', 'gbpusd', 'audusd', 'usdcad', 'usdchf', 'nzdusd',
                   'eurjpy', 'gbpjpy', 'eurgbp', 'audjpy', 'euraud', 'eurchf', 'audnzd']
    pip_size = [0.0001, 0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                0.01, 0.01, 0.0001, 0.01, 0.0001, 0.0001, 0.0001]
    pip_spread = [spread] * 14

    return instruments, pip_size, pip_spread


# def get_forex_18(spread):
#     instruments = ['usdcad', 'eurusd', 'usdchf', 'gbpusd', 'nzdusd', 'audusd', 'usdjpy', 'eurcad', 'euraud',
#                    'eurjpy', 'eurchf', 'eurgbp', 'audcad', 'gbpchf', 'gbpjpy', 'chfjpy', 'audjpy', 'audnzd']
#     pip_size = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.01, 0.0001, 0.0001, 0.01, 0.0001, 0.0001, 0.0001,
#                 0.0001, 0.01, 0.01, 0.01, 0.0001]
#     pip_spread = [spread] * 18
#
#     return instruments, pip_size, pip_spread


def get_forex_18(spread):
    instruments = ['eurusd', 'usdjpy', 'gbpusd', 'audusd', 'usdcad', 'usdchf', 'nzdusd',
                   'eurjpy', 'gbpjpy', 'eurgbp', 'audjpy', 'euraud', 'eurchf', 'audnzd',
                   'nzdjpy', 'gbpaud', 'gbpcad', 'eurnzd']
    pip_size = [0.0001, 0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                0.01, 0.01, 0.0001, 0.01, 0.0001, 0.0001, 0.0001,
                0.01, 0.0001, 0.0001, 0.0001]
    pip_spread = [spread] * 18

    return instruments, pip_size, pip_spread


def get_forex_26(spread):
    instruments = ['eurusd', 'usdjpy', 'gbpusd', 'audusd', 'usdcad', 'usdchf', 'nzdusd',
                   'eurjpy', 'gbpjpy', 'eurgbp', 'audjpy', 'euraud', 'eurchf', 'audnzd',
                   'nzdjpy', 'gbpcad', 'eurnzd', 'audcad', 'gbpchf', 'audchf', 'eurcad',
                   'cadjpy', 'cadchf', 'chfjpy', 'nzdcad', 'nzdchf']
    pip_size = [0.0001, 0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                0.01, 0.01, 0.0001, 0.01, 0.0001, 0.0001, 0.0001,
                0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                0.0001, 0.01, 0.0001, 0.0001, 0.01, 0.0001, 0.0001]
    pip_spread = [spread] * 28

    return instruments, pip_size, pip_spread


def get_forex_28(spread):
    instruments = ['eurusd', 'usdjpy', 'gbpusd', 'audusd', 'usdcad', 'usdchf', 'nzdusd',
                   'eurjpy', 'gbpjpy', 'eurgbp', 'audjpy', 'euraud', 'eurchf', 'audnzd',
                   'nzdjpy', 'gbpaud', 'gbpcad', 'eurnzd', 'audcad', 'gbpchf', 'audchf',
                   'eurcad', 'cadjpy', 'gbpnzd', 'cadchf', 'chfjpy', 'nzdcad', 'nzdchf']
    pip_size = [0.0001, 0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                0.01, 0.01, 0.0001, 0.01, 0.0001, 0.0001, 0.0001,
                0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                0.0001, 0.01, 0.0001, 0.0001, 0.01, 0.0001, 0.0001]
    pip_spread = [spread] * 28

    return instruments, pip_size, pip_spread


def get_forex_33(spread):
    instruments = ['eurusd', 'usdjpy', 'gbpusd', 'audusd', 'usdcad', 'usdchf', 'nzdusd',
                   'eurjpy', 'gbpjpy', 'eurgbp', 'audjpy', 'euraud', 'eurchf', 'audnzd',
                   'nzdjpy', 'gbpaud', 'gbpcad', 'eurnzd', 'audcad', 'gbpchf', 'audchf',
                   'eurcad', 'cadjpy', 'gbpnzd', 'cadchf', 'chfjpy', 'nzdcad', 'nzdchf',
                   'eurnok', 'eursek', 'usdmxn', 'usdnok', 'usdsek']
    pip_size = [0.0001, 0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                0.01, 0.01, 0.0001, 0.01, 0.0001, 0.0001, 0.0001,
                0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                0.0001, 0.01, 0.0001, 0.0001, 0.01, 0.0001, 0.0001,
                0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
    pip_spread = [spread] * 33

    return instruments, pip_size, pip_spread


def check_convert(account_currencies, instruments, lot_size, leverage, prices):
    v_account_currency = None
    for account_currency in account_currencies:
        v_convert = convert(account_currency, instruments, lot_size, leverage)
        v_check = False
        try:
            v_convert.calculate_margins_in_account_currency(prices)
            v_account_currency = account_currency
            v_check = True
            break
        except Exception as e:
            v_account_currency = None
            v_check = False

    return v_account_currency, v_check


def get_instruments_in_portfolio(num_of_instruments, num_of_instruments_in_portfolio, is_equal, spread):
    v_num_of_instruments = int(random.choice(num_of_instruments))

    if v_num_of_instruments == 4:
        v_instruments, v_pip_size, v_pip_spread = get_forex_7(spread)
    elif v_num_of_instruments == 7:
        v_instruments, v_pip_size, v_pip_spread = get_forex_7(spread)
    elif v_num_of_instruments == 12:
        v_instruments, v_pip_size, v_pip_spread = get_forex_12(spread)
    elif v_num_of_instruments == 14:
        v_instruments, v_pip_size, v_pip_spread = get_forex_14(spread)
    elif v_num_of_instruments == 18:
        v_instruments, v_pip_size, v_pip_spread = get_forex_18(spread)
    elif v_num_of_instruments == 26:
        v_instruments, v_pip_size, v_pip_spread = get_forex_26(spread)
    elif v_num_of_instruments == 28:
        v_instruments, v_pip_size, v_pip_spread = get_forex_28(spread)

    v_num_of_instruments_in_portfolio = int(random.choice(num_of_instruments_in_portfolio))

    v_pip_size_dict = {v_instruments[i]: v_pip_size[i] for i in range(len(v_instruments))}
    v_pip_spread_dict = {v_instruments[i]: v_pip_spread[i] for i in range(len(v_instruments))}

    if is_equal:
        v_instruments_in_portfolio = v_instruments
    else:
        v_instruments_in_portfolio = random.sample(v_instruments, v_num_of_instruments_in_portfolio)

    v_instruments_in_portfolio_sorted = sorted(v_instruments_in_portfolio)

    print(f'v_instruments_in_portfolio = {v_instruments_in_portfolio}')
    print(f'v_instruments_in_portfolio_sorted = {v_instruments_in_portfolio_sorted}')

    v_pip_size_in_portfolio = [v_pip_size_dict[instrument] for instrument in v_instruments_in_portfolio_sorted]
    v_pip_spread_in_portfolio = [v_pip_spread_dict[instrument] for instrument in v_instruments_in_portfolio_sorted]

    # v_instruments_in_portfolio, v_pip_size_in_portfolio, v_pip_spread_in_portfolio = zip(
    #     *random.sample(list(zip(v_instruments, v_pip_size, v_pip_spread)), v_num_of_instruments_in_portfolio))

    return v_instruments_in_portfolio_sorted, v_pip_size_in_portfolio, v_pip_spread_in_portfolio


def get_instruments_in_portfolio_fixed(instruments_in_portfolio, spread):
    v_instruments, v_pip_size, v_pip_spread = get_forex_28(spread)

    v_pip_size_dict = {v_instruments[i]: v_pip_size[i] for i in range(len(v_instruments))}
    v_pip_spread_dict = {v_instruments[i]: v_pip_spread[i] for i in range(len(v_instruments))}

    v_instruments_in_portfolio = instruments_in_portfolio
    v_instruments_in_portfolio_sorted = sorted(v_instruments_in_portfolio)

    print(f'v_instruments_in_portfolio = {v_instruments_in_portfolio}')
    print(f'v_instruments_in_portfolio_sorted = {v_instruments_in_portfolio_sorted}')

    v_pip_size_in_portfolio = [v_pip_size_dict[instrument] for instrument in v_instruments_in_portfolio_sorted]
    v_pip_spread_in_portfolio = [v_pip_spread_dict[instrument] for instrument in v_instruments_in_portfolio_sorted]

    # v_instruments_in_portfolio, v_pip_size_in_portfolio, v_pip_spread_in_portfolio = zip(
    #     *random.sample(list(zip(v_instruments, v_pip_size, v_pip_spread)), v_num_of_instruments_in_portfolio))

    return v_instruments_in_portfolio_sorted, v_pip_size_in_portfolio, v_pip_spread_in_portfolio
