import pprint

import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

accoounts_h1 = {
    'h1-1': '101-001-204423-001',
    'h1-2': '101-001-204423-002',
    'h1-3': '101-001-204423-003',
    'h1-4': '101-001-204423-005',
    'h1-5': '101-001-204423-006',
    'h1-6': '101-001-204423-015',
    'h1-7': '101-001-204423-016',
    'h1-8': '101-001-204423-017',
    'h1-9': '101-001-204423-018',
    'h1-10': '101-001-204423-019',
}

accoounts_d1 = {
    'd1-1': '101-001-204423-007',
    'd1-2': '101-001-204423-008',
    'd1-3': '101-001-204423-009',
    'd1-4': '101-001-204423-010',
    'd1-5': '101-001-204423-011',
    'd1-6': '101-001-204423-012',
    'd1-7': '101-001-204423-013',
    'd1-8': '101-001-204423-014',
    'd1-9': '101-001-204423-020',
    'd1-10': '101-001-204423-004',
}


def get_account_info(account_id):
    client = oandapyV20.API(access_token=oanda_access_token)
    r = accounts.AccountSummary(account_id)
    client.request(r)
    return r.response['account']


def main():
    total_balance_h1 = 0
    total_nav_h1 = 0
    total_balance_d1 = 0
    total_nav_d1 = 0

    for key, value in accoounts_h1.items():
        account_info = get_account_info(value)
        balance = account_info['balance']
        NAV = account_info['NAV']
        total_balance_h1 += float(balance)
        total_nav_h1 += float(NAV)
        pprint.pprint(f'Account: {key}')
        pprint.pprint(f'Balance: {balance}, NAV: {NAV}')
        #   pprint.pprint(f'Info: {account_info}')
        print()

    for key, value in accoounts_d1.items():
        account_info = get_account_info(value)
        balance = account_info['balance']
        NAV = account_info['NAV']
        total_balance_d1 += float(balance)
        total_nav_d1 += float(NAV)
        pprint.pprint(f'Account: {key}')
        pprint.pprint(f'Balance: {balance}, NAV: {NAV}')
        #   pprint.pprint(f'Info: {account_info}')
        print()

    print('--------------------------------')
    print(f'Total Balance H1: {total_balance_h1}, Total Nav H1: {total_nav_h1}')
    print(f'Total Balance D1: {total_balance_d1}, Total Nav D1: {total_nav_d1}')


if __name__ == "__main__":
    main()
