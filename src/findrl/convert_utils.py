#   https://www.fxcm.com/markets/insights/how-is-the-value-of-a-pip-determined/
#   https://www.fxcm.com/markets/insights/pip-calculator/

class convert:

    def __init__(self,
                 account_currency,
                 instruments,
                 lot_size,
                 leverage):

        self.account_currency = account_currency
        self.instruments = instruments

        if lot_size.lower() == 'nano':
            self.lot_size = 100
        elif lot_size.lower() == 'micro':
            self.lot_size = 1000
        elif lot_size.lower() == 'mini':
            self.lot_size = 10000
        elif lot_size.lower() == 'standard':
            self.lot_size = 100000

        self.leverage = leverage

    def calculate_margins_in_account_currency(self, current_prices):

        margin_values = []

        # dictionary to keep prices for each currency, assuming that current_prices has prices in the same order as instruments list has instument names
        current_prices_dict = {}

        instrument_index = 0
        for instrument in self.instruments:
            current_prices_dict[instrument] = current_prices[instrument_index]
            instrument_index += 1

        m = 0
        for instrument in self.instruments:
            first_currency = instrument[0:3]
            second_currency = instrument[4:7]

            #   print(first_currency + "/" + second_currency)

            # counter currency same as account currency
            if second_currency == self.account_currency:
                margin_value = current_prices[m]
            # base currency same as account currency
            elif first_currency == self.account_currency:
                margin_value = 1
                # none of the currency pair is the same as account currency
            # is needed the currency rate for the counter currency/account currency
            else:
                ##base currency/account currency rate is retrieved from stored values in dictionary
                if first_currency + self.account_currency in self.instruments:
                    base_account_rate = current_prices_dict[first_currency + self.account_currency]
                elif self.account_currency + first_currency in self.instruments:
                    base_account_rate = 1 / current_prices_dict[self.account_currency + first_currency]

                margin_value = base_account_rate

            margin_values.append(margin_value * self.lot_size / self.leverage)

            m += 1

        return margin_values

    def calculate_pip_value_in_account_currency(self, current_prices):

        pip_values = []

        # dictionary to keep prices for each currency, assuming that current_prices has prices in the same order as instruments list has instument names
        current_prices_dict = {}

        instrument_index = 0
        for instrument in self.instruments:
            current_prices_dict[instrument] = current_prices[instrument_index]
            instrument_index += 1

        # account currency is usd
        m = 0
        for instrument in self.instruments:
            first_currency = instrument[0:3]
            second_currency = instrument[4:7]

            # counter currency same as account currency
            if second_currency == self.account_currency:
                pip_value = 0.0001
            # base currency same as account currency
            elif first_currency == self.account_currency:
                # counter currency is jpy
                if second_currency == 'jpy':
                    pip_value = 0.01 / current_prices[m]
                # counter currency is not jpy
                else:
                    pip_value = 0.0001 / current_prices[m]
                # none of the currency pair is the same as account currency
            # is needed the currency rate for the base currency/account currency
            else:
                ##base currency/account currency rate is retrieved from stored values in dictionary
                if first_currency + self.account_currency in self.instruments:
                    base_account_rate = current_prices_dict[first_currency + self.account_currency]
                elif self.account_currency + first_currency in self.instruments:
                    base_account_rate = 1 / current_prices_dict[self.account_currency + first_currency]

                if second_currency == 'jpy':
                    pip_value = base_account_rate * 0.01 / current_prices[m]
                else:
                    pip_value = base_account_rate * 0.0001 / current_prices[m]

            pip_values.append(pip_value * self.lot_size)

            m += 1

        return pip_values
