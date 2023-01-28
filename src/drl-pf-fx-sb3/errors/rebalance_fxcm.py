import datetime
import time

import fxcmpy
import numpy as np

TIME_TO_SLEEP = 0  # 0 - real, 5 - zulutradea demo


def get_open_positions_long(instrument, open_positions):
    return [element['amountK'] for element in open_positions if
            element['currency'] == instrument and element['isBuy']]


def get_open_positions_short(instrument, open_positions):
    return [element['amountK'] for element in open_positions if
            element['currency'] == instrument and not element['isBuy']]


def get_open_positions_net(instrument, open_positions):
    total_open_positions_long = 0
    total_open_positions_short = 0
    for element in get_open_positions_long(instrument, open_positions):
        total_open_positions_long += element
    for element in get_open_positions_short(instrument, open_positions):
        total_open_positions_short += element

    return total_open_positions_long - total_open_positions_short


#   convert('USD', self.instruments, self.lot_size, self.leverage)
def calculate_margin_in_account_currency(instruments, lot_size, leverage, account_currency, current_prices):
    if lot_size.lower() == 'nano':
        lot_size = 100
    elif lot_size.lower() == 'micro':
        lot_size = 1000
    elif lot_size.lower() == 'mini':
        lot_size = 10000
    elif lot_size.lower() == 'standard':
        lot_size = 100000

    margin_values = []

    # dictionary to keep prices for each currency, assuming that current_prices has prices in the same order as instruments list has instument names
    prices_for_currency = {}

    instrument_index = 0
    for instrument in instruments:
        prices_for_currency[instrument] = current_prices[instrument_index]
        instrument_index += 1

    if account_currency == 'USD':
        m = 0
        for instrument in instruments:
            first_currency = instrument[0:3]
            second_currency = instrument[4:7]

            #   print(first_currency + "/" + second_currency)

            # counter currency same as account currency
            if second_currency == 'USD':
                margin_value = current_prices[m]
            # base currency same as account currency
            elif first_currency == 'USD':
                margin_value = 1
                # none of the currency pair is the same as account currency
            # is needed the currency rate for the counter currency/account currency
            else:
                ##base currency/account currency rate is retrieved from stored values in dictionary
                if first_currency + "USD" in instruments:
                    base_account_rate = prices_for_currency[first_currency + "USD"]
                elif "USD" + first_currency in instruments:
                    base_account_rate = 1 / prices_for_currency["USD" + first_currency]

                margin_value = base_account_rate

            margin_values.append(margin_value * lot_size / leverage)

            m += 1

    return margin_values


def rebalance_fxcm(fxcm_access_token, instruments, target_weights):
    now = datetime.datetime.now()
    print(f'Start rebalance date and time :{now.strftime("%Y-%m-%d %H:%M:%S")}')

    con = fxcmpy.fxcmpy(access_token=fxcm_access_token, server='demo', log_level='error')
    accounts = con.get_accounts()
    equity = accounts['equity'][0]

    print(f'equity: {equity}')

    current_prices = []

    for instrument in instruments:
        con.subscribe_market_data(instrument)
        current_prices.append(con.get_last_price(instrument))

    for instrument in instruments:
        con.unsubscribe_market_data(instrument)

    target_portfolio_values = np.multiply(equity, target_weights)
    current_margins_per_lot = calculate_margin_in_account_currency(current_prices)
    target_positions = np.around(np.divide(target_portfolio_values[:-1], current_margins_per_lot), 2) * 100

    current_positions_net = []

    open_positions = con.get_open_positions(kind='list')
    if len(open_positions) == 0:
        current_positions_net = np.zeros(len(instruments))
    else:
        for instrument in instruments:
            current_position_net = get_open_positions_net(instrument, open_positions)
            current_positions_net.append(current_position_net)

    current_positions_net = np.asarray(current_positions_net)
    target_positions = np.asarray(target_positions)
    #   trade_amount = np.round(target_positions - current_positions_net)
    trade_amount = np.subtract(target_positions, current_positions_net)

    #   trade_amount = np.floor(trade_amount*100)/100

    print(f'Current positions net: {current_positions_net}')
    print(f'Target positions: {target_positions}')
    print(f"Trade amount: {trade_amount}")

    m = 0
    for instrument in instruments:
        if target_positions[m] == 0:
            con.close_all_for_symbol(instrument)
            print('Close all  ' + instrument)
        elif target_positions[m] > 0 and trade_amount[m] == 0:
            pass
        elif target_positions[m] > 0 and trade_amount[m] > 0:
            order = None
            while order is None:
                order = con.create_market_buy_order(instrument, np.abs(trade_amount[m]))
            print('Long ' + instrument + ' ' + str(np.abs(trade_amount[m])))
        elif target_positions[m] > 0 and trade_amount[m] < 0:
            con.close_all_for_symbol(instrument)
            time.sleep(TIME_TO_SLEEP)  # Wait for 5 seconds zulutradea demo account
            order = None
            while order is None:
                order = con.create_market_buy_order(instrument, np.abs(target_positions[m]))
            print('Adjust ' + instrument + ' ' + str(np.abs(target_positions[m])))
        elif target_positions[m] < 0 and trade_amount[m] == 0:
            pass
        elif target_positions[m] < 0 and trade_amount[m] < 0:
            order = None
            while order is None:
                order = con.create_market_sell_order(instrument, np.abs(trade_amount[m]))
            print('Short ' + instrument + ' ' + str(np.abs(trade_amount[m])))
        elif target_positions[m] < 0 and trade_amount[m] > 0:
            con.close_all_for_symbol(instrument)
            time.sleep(TIME_TO_SLEEP)
            order = None
            while order is None:
                order = con.create_market_sell_order(instrument, np.abs(target_positions[m]))
            print('Adjust ' + instrument + ' ' + str(np.abs(target_positions[m])))
        m += 1
        time.sleep(TIME_TO_SLEEP)

    con.close()

    now = datetime.datetime.now()
    print(f'End rebalance date and time :{now.strftime("%Y-%m-%d %H:%M:%S")}')
