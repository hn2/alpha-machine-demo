import datetime
import time

import fxcmpy
import numpy as np

from .convert_utils import convert

TIME_TO_SLEEP = 1  # 0 - real, 5 - zulutradea demo


def _get_open_positions_long(instrument, open_positions):
    return [element['amountK'] for element in open_positions if
            element['currency'] == instrument and element['isBuy']]


def _get_open_positions_short(instrument, open_positions):
    return [element['amountK'] for element in open_positions if
            element['currency'] == instrument and not element['isBuy']]


def _get_open_positions_net(instrument, open_positions):
    total_open_positions_long = 0
    total_open_positions_short = 0
    for element in _get_open_positions_long(instrument, open_positions):
        total_open_positions_long += element
    for element in _get_open_positions_short(instrument, open_positions):
        total_open_positions_short += element

    return total_open_positions_long - total_open_positions_short


def _get_trade_id_for_instrument(conn, currency):
    open_positions = conn.get_open_positions(kind='dataframe')
    trade_id = open_positions.loc[open_positions['currency'] == currency]['tradeId']

    return trade_id


def _get_last_prices(conn, instruments):
    current_prices = []

    for instrument in instruments:
        conn.subscribe_market_data(instrument)
        last_price = conn.get_last_price(instrument)[['Bid', 'Ask']].mean()
        current_prices.append(last_price)
        conn.unsubscribe_market_data(instrument)

    return current_prices


#   order = con.open_trade(symbol=’EUR/USD’, is_buy=True, amount=6, order_type=’MarketRange’, time_in_force=’IOC’, at_market=3, stop=1.02, limit=1.04, is_in_pips=False)
def _create_market_range_buy_order(conn, instrument, amount, rate):
    #   100 pips stop based on MAE
    # if 'JPY' in instrument:
    #     stop = rate - 1
    # else:
    #     stop = rate - 0.01
    if 'JPY' in instrument:
        stop = rate - 5
    else:
        stop = rate - 0.05
    order_id = conn.open_trade(symbol=instrument, is_buy=True, amount=amount,
                               time_in_force='IOC', order_type='MarketRange',
                               limit=1.1 * rate, stop=stop,
                               is_in_pips=False, rate=rate, at_market=5, account_id=None)

    return order_id


def _create_market_range_sell_order(conn, instrument, amount, rate):
    #   100 pips stop based on MAE
    # if 'JPY' in instrument:
    #     stop = rate + 1
    # else:
    #     stop = rate + 0.01
    if 'JPY' in instrument:
        stop = rate + 5
    else:
        stop = rate + 0.05
    order_id = conn.open_trade(symbol=instrument, is_buy=False, amount=amount,
                               time_in_force='IOC', order_type='MarketRange',
                               limit=0.9 * rate, stop=stop,
                               is_in_pips=False, rate=rate, at_market=5, account_id=None)

    return order_id


def _create_limit_buy_order(conn, instrument, amount, rate):
    order_id = conn.create_entry_order(symbol=instrument, is_buy=True, amount=amount,
                                       time_in_force='GTC', order_type='Entry', limit=1.001 * rate,
                                       is_in_pips=False,
                                       rate=rate, stop=0.9 * rate, trailing_step=None, trailing_stop_step=None,
                                       order_range=None, expiration=None, account_id=None)

    return order_id


def _create_limit_sell_order(conn, instrument, amount, rate):
    order_id = conn.create_entry_order(symbol=instrument, is_buy=False, amount=amount,
                                       time_in_force='GTC', order_type='Entry', limit=0.999 * rate,
                                       is_in_pips=False,
                                       rate=rate, stop=1.1 * rate, trailing_step=None, trailing_stop_step=None,
                                       order_range=None, expiration=None, account_id=None)

    return order_id


def delete_pending_orders(fxcm_access_token, server, trade):
    conn = fxcmpy.fxcmpy(access_token=fxcm_access_token, server=server, log_level='error')

    if trade:

        order_ids = conn.get_order_ids()
        for order_id in order_ids:
            conn.delete_order(order_id)

    conn.close()


def _rebalance_market(fxcm_access_token, server, instruments, target_weights, lot_size, leverage, account_currency,
                      trade):
    conn = fxcmpy.fxcmpy(access_token=fxcm_access_token, server=server, log_level='error')

    #   _liquidate(conn, instruments, trade)

    accounts = conn.get_accounts()
    equity = accounts['equity'][0]
    account_currency = account_currency.upper()

    print(f'equity: {equity}')

    last_prices = _get_last_prices(conn, instruments)

    #   print(last_prices)

    my_convert = convert(account_currency, [instrument[0:3] + instrument[4:7] for instrument in instruments], lot_size,
                         leverage)

    target_weights = np.asarray(target_weights, dtype='float32')

    #   print(target_weights)

    target_portfolio_values = np.multiply(equity, target_weights)

    print(f'Target_portfolio_values: {target_portfolio_values}')

    current_prices = last_prices

    current_pip_value_in_account_currency = my_convert.calculate_pip_value_in_account_currency(last_prices)

    current_margins_per_lot = my_convert.calculate_margins_in_account_currency(last_prices)

    target_positions_prices = np.around(np.divide(target_portfolio_values, last_prices), 2)
    target_positions_pip_value_in_account_currency = np.around(
        np.divide(target_portfolio_values, current_pip_value_in_account_currency), 2)
    target_positions_margins_per_lot = np.around(np.divide(target_portfolio_values, current_margins_per_lot), 2)

    if account_currency.lower() == 'jpy':
        target_positions_margins_per_lot = np.multiply(target_positions_margins_per_lot, 100)

    current_positions = []

    open_positions = conn.get_open_positions(kind='list')
    if len(open_positions) == 0:
        current_positions = np.zeros(len(instruments))
    else:
        for instrument in instruments:
            current_position_net = _get_open_positions_net(instrument, open_positions)
            current_positions.append(current_position_net)

    current_positions = (np.asarray(current_positions)).astype(int)
    target_positions = (np.around(np.asarray(target_positions_margins_per_lot))).astype(int)
    trade_amount = (np.subtract(target_positions, current_positions)).astype(int)

    print(f'Current positions net: {current_positions}')
    print(f'Target positions: {target_positions}')
    print(f'Trade amount: {trade_amount}')

    # print(f'Deleting pending orders...')
    # _delete_all_pending_orders(conn)

    if trade:
        now = datetime.datetime.now()
        print(f'Start rebalance date and time :{now.strftime("%Y-%m-%d %H:%M:%S")}')

        for i, instrument in enumerate(instruments):

            print(f'Instrument is {instrument}')
            print(f'Current positions is {current_positions[i]}')
            print(f'Target positions is {target_positions[i]}')
            print(f'Trade positions is {trade_amount[i]}')

            if target_positions[i] == current_positions[i]:
                continue
            elif target_positions[i] == 0:
                conn.close_all_for_symbol(instrument)
                continue
            elif current_positions[i] >= 0 and target_positions[i] >= 0 and trade_amount[i] > 0:
                order_id = None
                while order_id is None:
                    order_id = conn.create_market_buy_order(instrument, int(np.abs(trade_amount[i])))
                print(f'Buy {np.abs(trade_amount[i])} {instrument}')
                continue
            elif current_positions[i] >= 0 and target_positions[i] >= 0 and trade_amount[i] < 0:
                trade_id_for_instrument = _get_trade_id_for_instrument(conn, instrument)
                if trade_id_for_instrument.count() == 1:
                    conn.close_trade(int(trade_id_for_instrument), int(np.abs(trade_amount[i])))
                else:
                    conn.close_all_for_symbol(instrument)
                    time.sleep(TIME_TO_SLEEP)
                    order_id = None
                    while order_id is None:
                        order_id = conn.create_market_buy_order(instrument, int(np.abs(target_positions[i])))
                print(f'Adjust position {np.abs(trade_amount[i])} {instrument}')
                continue
            elif current_positions[i] <= 0 and target_positions[i] <= 0 and trade_amount[i] < 0:
                order_id = None
                while order_id is None:
                    order_id = conn.create_market_sell_order(instrument, int(np.abs(trade_amount[i])))
                print(f'Sell {np.abs(trade_amount[i])} {instrument}')
                continue
            elif current_positions[i] <= 0 and target_positions[i] <= 0 and trade_amount[i] > 0:
                trade_id_for_instrument = _get_trade_id_for_instrument(conn, instrument)
                if trade_id_for_instrument.count() == 1:
                    conn.close_trade(int(trade_id_for_instrument), int(np.abs(trade_amount[i])))
                else:
                    conn.close_all_for_symbol(instrument)
                    time.sleep(TIME_TO_SLEEP)
                    order_id = None
                    while order_id is None:
                        order_id = conn.create_market_sell_order(instrument, int(np.abs(target_positions[i])))
                print(f'Adjust position {np.abs(trade_amount[i])} {instrument}')
                continue
            elif current_positions[i] < 0 and target_positions[i] > 0:
                conn.close_all_for_symbol(instrument)
                time.sleep(TIME_TO_SLEEP)
                order_id = None
                while order_id is None:
                    order_id = conn.create_market_buy_order(instrument, int(np.abs(target_positions[i])))
                print(f'Close short and buy {np.abs(target_positions[i])} {instrument}')
                continue
            elif current_positions[i] > 0 and target_positions[i] < 0:
                conn.close_all_for_symbol(instrument)
                time.sleep(TIME_TO_SLEEP)
                order_id = None
                while order_id is None:
                    order_id = conn.create_market_sell_order(instrument, int(np.abs(target_positions[i])))
                print(f'Close long and sell {np.abs(target_positions[i])} {instrument}')
                continue
            time.sleep(TIME_TO_SLEEP)

        now = datetime.datetime.now()
        print(f'End rebalance date and time :{now.strftime("%Y-%m-%d %H:%M:%S")}')

    conn.close()

    return current_prices, current_pip_value_in_account_currency, current_margins_per_lot, \
           target_positions_prices, target_positions_pip_value_in_account_currency, \
           target_positions_margins_per_lot, current_positions, target_positions, trade_amount


def _rebalance_market_range(fxcm_access_token, server, instruments, target_weights, lot_size, leverage,
                            account_currency, trade):
    conn = fxcmpy.fxcmpy(access_token=fxcm_access_token, server=server, log_level='error')

    #   _liquidate(conn, instruments, trade)

    accounts = conn.get_accounts()
    equity = accounts['equity'][0]
    account_currency = account_currency.upper()

    print(f'equity: {equity}')

    last_prices = _get_last_prices(conn, instruments)

    #   print(last_prices)

    my_convert = convert(account_currency, [instrument[0:3] + instrument[4:7] for instrument in instruments],
                         lot_size, leverage)

    target_weights = np.asarray(target_weights, dtype='float32')

    #   print(target_weights)

    target_portfolio_values = np.multiply(equity, target_weights)

    print(f'Target_portfolio_values: {target_portfolio_values}')

    current_prices = last_prices

    current_pip_value_in_account_currency = my_convert.calculate_pip_value_in_account_currency(last_prices)

    current_margins_per_lot = my_convert.calculate_margins_in_account_currency(last_prices)

    target_positions_prices = np.around(np.divide(target_portfolio_values, last_prices), 2)
    target_positions_pip_value_in_account_currency = np.around(
        np.divide(target_portfolio_values, current_pip_value_in_account_currency), 2)
    target_positions_margins_per_lot = np.around(np.divide(target_portfolio_values, current_margins_per_lot), 2)

    if account_currency.lower() == 'jpy':
        target_positions_margins_per_lot = np.multiply(target_positions_margins_per_lot, 100)

    current_positions = []

    open_positions = conn.get_open_positions(kind='list')
    if len(open_positions) == 0:
        current_positions = np.zeros(len(instruments))
    else:
        for instrument in instruments:
            current_position_net = _get_open_positions_net(instrument, open_positions)
            current_positions.append(current_position_net)

    current_positions = (np.asarray(current_positions)).astype(int)
    target_positions = (np.around(np.asarray(target_positions_margins_per_lot))).astype(int)
    trade_amount = (np.subtract(target_positions, current_positions)).astype(int)

    print(f'Current positions net: {current_positions}')
    print(f'Target positions: {target_positions}')
    print(f'Trade amount: {trade_amount}')

    # print(f'Deleting pending orders...')
    # _delete_all_pending_orders(conn)

    if trade:
        now = datetime.datetime.now()
        print(f'Start rebalance date and time :{now.strftime("%Y-%m-%d %H:%M:%S")}')

        for i, instrument in enumerate(instruments):

            print(f'Instrument is {instrument}')
            print(f'Current positions is {current_positions[i]}')
            print(f'Target positions is {target_positions[i]}')
            print(f'Trade positions is {trade_amount[i]}')

            if target_positions[i] == current_positions[i]:
                continue
            elif target_positions[i] == 0:
                conn.close_all_for_symbol(instrument)
                continue
            elif current_positions[i] >= 0 and target_positions[i] >= 0 and trade_amount[i] > 0:
                order_id = None
                while order_id is None:
                    # order = conn.create_market_buy_order(instrument, int(np.abs(trade_amount[i])))
                    order_id = _create_market_range_buy_order(conn, instrument, int(np.abs(trade_amount[i])),
                                                              last_prices[i])
                print(f'Buy {np.abs(trade_amount[i])} {instrument}')
                continue
            elif current_positions[i] >= 0 and target_positions[i] >= 0 and trade_amount[i] < 0:
                trade_id_for_instrument = _get_trade_id_for_instrument(conn, instrument)
                if trade_id_for_instrument.count() == 1:
                    conn.close_trade(int(trade_id_for_instrument), int(np.abs(trade_amount[i])))
                else:
                    conn.close_all_for_symbol(instrument)
                    time.sleep(TIME_TO_SLEEP)
                    order_id = None
                    while order_id is None:
                        # order = conn.create_market_buy_order(instrument, int(np.abs(target_positions[i])))
                        order_id = _create_market_range_buy_order(conn, instrument, int(np.abs(target_positions[i])),
                                                                  last_prices[i])
                print(f'Adjust position {np.abs(trade_amount[i])} {instrument}')
                continue
            elif current_positions[i] <= 0 and target_positions[i] <= 0 and trade_amount[i] < 0:
                order_id = None
                while order_id is None:
                    #   order = conn.create_market_sell_order(instrument, int(np.abs(trade_amount[i])))
                    order_id = _create_market_range_sell_order(conn, instrument, int(np.abs(trade_amount[i])),
                                                               last_prices[i])
                print(f'Sell {np.abs(trade_amount[i])} {instrument}')
                continue
            elif current_positions[i] <= 0 and target_positions[i] <= 0 and trade_amount[i] > 0:
                trade_id_for_instrument = _get_trade_id_for_instrument(conn, instrument)
                if trade_id_for_instrument.count() == 1:
                    conn.close_trade(int(trade_id_for_instrument), int(np.abs(trade_amount[i])))
                else:
                    conn.close_all_for_symbol(instrument)
                    time.sleep(TIME_TO_SLEEP)
                    order_id = None
                    while order_id is None:
                        # order = conn.create_market_sell_order(instrument, int(np.abs(target_positions[i])))
                        order_id = _create_market_range_sell_order(conn, instrument, int(np.abs(target_positions[i])),
                                                                   last_prices[i])
                print(f'Adjust position {np.abs(trade_amount[i])} {instrument}')
                continue
            elif current_positions[i] < 0 and target_positions[i] > 0:
                conn.close_all_for_symbol(instrument)
                time.sleep(TIME_TO_SLEEP)
                order_id = None
                while order_id is None:
                    # order = conn.create_market_buy_order(instrument, int(np.abs(target_positions[i])))
                    order_id = _create_market_range_buy_order(conn, instrument, int(np.abs(trade_amount[i])),
                                                              last_prices[i])
                print(f'Close short and buy {np.abs(target_positions[i])} {instrument}')
                continue
            elif current_positions[i] > 0 and target_positions[i] < 0:
                conn.close_all_for_symbol(instrument)
                time.sleep(TIME_TO_SLEEP)
                order_id = None
                while order_id is None:
                    # order = conn.create_market_sell_order(instrument, int(np.abs(target_positions[i])))
                    order_id = _create_market_range_sell_order(conn, instrument, int(np.abs(target_positions[i])),
                                                               last_prices[i])
                print(f'Close long and sell {np.abs(target_positions[i])} {instrument}')
                continue
            time.sleep(TIME_TO_SLEEP)

        now = datetime.datetime.now()
        print(f'End rebalance date and time :{now.strftime("%Y-%m-%d %H:%M:%S")}')

    conn.close()

    return current_prices, current_pip_value_in_account_currency, current_margins_per_lot, \
           target_positions_prices, target_positions_pip_value_in_account_currency, \
           target_positions_margins_per_lot, current_positions, target_positions, trade_amount


def _rebalance_limit(fxcm_access_token, server, instruments, target_weights, lot_size, leverage, account_currency,
                     trade):
    conn = fxcmpy.fxcmpy(access_token=fxcm_access_token, server=server, log_level='error')

    #   _liquidate(conn, instruments, trade)

    accounts = conn.get_accounts()
    equity = accounts['equity'][0]
    account_currency = account_currency.upper()

    print(f'equity: {equity}')

    last_prices = _get_last_prices(conn, instruments)

    #   print(last_prices)

    my_convert = convert(account_currency, [instrument[0:3] + instrument[4:7] for instrument in instruments],
                         lot_size, leverage)

    target_weights = np.asarray(target_weights, dtype='float32')

    #   print(target_weights)

    target_portfolio_values = np.multiply(equity, target_weights)

    print(f'Target_portfolio_values: {target_portfolio_values}')

    current_prices = last_prices

    current_pip_value_in_account_currency = my_convert.calculate_pip_value_in_account_currency(last_prices)

    current_margins_per_lot = my_convert.calculate_margins_in_account_currency(last_prices)

    target_positions_prices = np.around(np.divide(target_portfolio_values, last_prices), 2)
    target_positions_pip_value_in_account_currency = np.around(
        np.divide(target_portfolio_values, current_pip_value_in_account_currency), 2)
    target_positions_margins_per_lot = np.around(np.divide(target_portfolio_values, current_margins_per_lot), 2)

    if account_currency.lower() == 'jpy':
        target_positions_margins_per_lot = np.multiply(target_positions_margins_per_lot, 100)

    current_positions = []

    open_positions = conn.get_open_positions(kind='list')
    if len(open_positions) == 0:
        current_positions = np.zeros(len(instruments))
    else:
        for instrument in instruments:
            current_position_net = _get_open_positions_net(instrument, open_positions)
            current_positions.append(current_position_net)

    current_positions = (np.asarray(current_positions)).astype(int)
    target_positions = (np.around(np.asarray(target_positions_margins_per_lot))).astype(int)
    trade_amount = (np.subtract(target_positions, current_positions)).astype(int)

    print(f'Current positions net: {current_positions}')
    print(f'Target positions: {target_positions}')
    print(f'Trade amount: {trade_amount}')

    # print(f'Deleting pending orders...')
    # _delete_all_pending_orders(conn)

    if trade:
        now = datetime.datetime.now()
        print(f'Start rebalance date and time :{now.strftime("%Y-%m-%d %H:%M:%S")}')

        for i, instrument in enumerate(instruments):

            print(f'Instrument is {instrument}')
            print(f'Current positions is {current_positions[i]}')
            print(f'Target positions is {target_positions[i]}')
            print(f'Trade positions is {trade_amount[i]}')

            if target_positions[i] == current_positions[i]:
                continue
            elif target_positions[i] == 0:
                conn.close_all_for_symbol(instrument)
                continue
            elif current_positions[i] >= 0 and target_positions[i] >= 0 and trade_amount[i] > 0:
                order_id = None
                while order_id is None:
                    # order = conn.create_market_buy_order(instrument, int(np.abs(trade_amount[i])))
                    order_id = _create_limit_buy_order(conn, instrument, int(np.abs(trade_amount[i])), last_prices[i])
                print(f'Buy {np.abs(trade_amount[i])} {instrument}')
                continue
            elif current_positions[i] >= 0 and target_positions[i] >= 0 and trade_amount[i] < 0:
                trade_id_for_instrument = _get_trade_id_for_instrument(conn, instrument)
                if trade_id_for_instrument.count() == 1:
                    conn.close_trade(int(trade_id_for_instrument), int(np.abs(trade_amount[i])))
                else:
                    conn.close_all_for_symbol(instrument)
                    time.sleep(TIME_TO_SLEEP)
                    order_id = None
                    while order_id is None:
                        # order = conn.create_market_buy_order(instrument, int(np.abs(target_positions[i])))
                        order_id = _create_limit_buy_order(conn, instrument, int(np.abs(target_positions[i])),
                                                           last_prices[i])
                print(f'Adjust position {np.abs(trade_amount[i])} {instrument}')
                continue
            elif current_positions[i] <= 0 and target_positions[i] <= 0 and trade_amount[i] < 0:
                order_id = None
                while order_id is None:
                    #   order = conn.create_market_sell_order(instrument, int(np.abs(trade_amount[i])))
                    order_id = _create_limit_sell_order(conn, instrument, int(np.abs(trade_amount[i])), last_prices[i])
                print(f'Sell {np.abs(trade_amount[i])} {instrument}')
                continue
            elif current_positions[i] <= 0 and target_positions[i] <= 0 and trade_amount[i] > 0:
                trade_id_for_instrument = _get_trade_id_for_instrument(conn, instrument)
                if trade_id_for_instrument.count() == 1:
                    conn.close_trade(int(trade_id_for_instrument), int(np.abs(trade_amount[i])))
                else:
                    conn.close_all_for_symbol(instrument)
                    time.sleep(TIME_TO_SLEEP)
                    order_id = None
                    while order_id is None:
                        # order = conn.create_market_sell_order(instrument, int(np.abs(target_positions[i])))
                        order_id = _create_limit_sell_order(conn, instrument, int(np.abs(target_positions[i])),
                                                            last_prices[i])
                print(f'Adjust position {np.abs(trade_amount[i])} {instrument}')
                continue
            elif current_positions[i] < 0 and target_positions[i] > 0:
                conn.close_all_for_symbol(instrument)
                time.sleep(TIME_TO_SLEEP)
                order_id = None
                while order_id is None:
                    # order = conn.create_market_buy_order(instrument, int(np.abs(target_positions[i])))
                    order_id = _create_limit_buy_order(conn, instrument, int(np.abs(trade_amount[i])), last_prices[i])
                print(f'Close short and buy {np.abs(target_positions[i])} {instrument}')
                continue
            elif current_positions[i] > 0 and target_positions[i] < 0:
                conn.close_all_for_symbol(instrument)
                time.sleep(TIME_TO_SLEEP)
                order_id = None
                while order_id is None:
                    # order = conn.create_market_sell_order(instrument, int(np.abs(target_positions[i])))
                    order_id = _create_limit_sell_order(conn, instrument, int(np.abs(target_positions[i])),
                                                        last_prices[i])
                print(f'Close long and sell {np.abs(target_positions[i])} {instrument}')
                continue
            time.sleep(TIME_TO_SLEEP)

        now = datetime.datetime.now()
        print(f'End rebalance date and time :{now.strftime("%Y-%m-%d %H:%M:%S")}')

    conn.close()

    return current_prices, current_pip_value_in_account_currency, current_margins_per_lot, \
           target_positions_prices, target_positions_pip_value_in_account_currency, \
           target_positions_margins_per_lot, current_positions, target_positions, trade_amount


def liquidate(fxcm_access_token, server, instruments, trade):
    conn = fxcmpy.fxcmpy(access_token=fxcm_access_token, server=server, log_level='error')

    open_positions = conn.get_open_positions(kind='dataframe')
    instruments_invested = []
    for index, row in open_positions.iterrows():
        instruments_invested.append(row['currency'])

    #   instruments_invested = set(instruments_invested)

    instruments_to_liquidate = [x for x in instruments_invested if x not in instruments]

    if instruments_to_liquidate and trade:

        now = datetime.datetime.now()
        print(f'Start liquidate date and time :{now.strftime("%Y-%m-%d %H:%M:%S")}')

        for i, instrument in enumerate(instruments_to_liquidate):
            print(f'Instrument to liquidate is {instrument}')
            conn.close_all_for_symbol(instrument)
            time.sleep(TIME_TO_SLEEP)

        now = datetime.datetime.now()
        print(f'End liquidate date and time :{now.strftime("%Y-%m-%d %H:%M:%S")}')

    conn.close()

    return


def rebalance(fxcm_access_token, server, instruments, target_weights, lot_size, leverage, account_currency, trade,
              order_type):
    if order_type.lower() == 'market':
        _rebalance_market(fxcm_access_token, server, instruments, target_weights, lot_size, leverage, account_currency,
                          trade)
    if order_type.lower() == 'market-range':
        _rebalance_market_range(fxcm_access_token, server, instruments, target_weights, lot_size, leverage,
                                account_currency, trade)
    elif order_type.lower() == 'limit':
        _rebalance_limit(fxcm_access_token, server, instruments, target_weights, lot_size, leverage, account_currency,
                         trade)
