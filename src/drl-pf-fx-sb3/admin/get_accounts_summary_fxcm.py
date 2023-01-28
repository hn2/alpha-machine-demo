import time

import fxcmpy
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

FXCM_ACCESS_TOKEN_REAL_1 = '0eb2ab9a2ae56b3d39f2e5a1da29ed46eeaa94ce'  # real acct no# 98069267
FXCM_ACCESS_TOKEN_REAL_2 = 'd8f85101a293b514a5be3022b5977295a4ac5043'  # real acct no# 98069135
FXCM_ACCESS_TOKEN_DEMO_1 = 'c29a91dec8e633fc85c44d32b94e7172369a8fca'  # demo acct no#  51580155
FXCM_ACCESS_TOKEN_DEMO_2 = '2bcee0bdadb457e21c5313d817a4f6bad634f6ec'  # demo acct no#  51582057
FXCM_ACCESS_TOKEN_DEMO_3 = '2f645c34b2be58fea00a4753036ec0d2e04108c4'  # demo acct no#  51582140
FXCM_ACCESS_TOKEN_DEMO_4 = '6310fa7f0f792856a5b3dc816f7900b3cadd37fc'  # demo acct no#  51582126
FXCM_ACCESS_TOKEN_DEMO_5 = '229c7c1e262fe2d74c1c50202fc0c6edd440633a'  # demo acct no#  51582141
FXCM_ACCESS_TOKEN_DEMO_6 = '0de34c687ac67ef187f09a6db1a20c3eb5c447d7'  # demo acct no#  51582129
FXCM_ACCESS_TOKEN_DEMO_7 = '76b984c408afa77c9dc532f7e71ccf2e28629e82'  # demo acct no#  51582130
FXCM_ACCESS_TOKEN_DEMO_8 = '7b67de2cfc40e19d8742c179a719ba6e2f0b137e'  # demo acct no#  51582142
FXCM_ACCESS_TOKEN_DEMO_9 = '7f5ecb274430cb582a247794f4dd7817cb975563'  # demo acct no#  51582132
FXCM_ACCESS_TOKEN_DEMO_10 = '34c017204c0dc693e01ba10105e02e5d792c7f69'  # demo acct no#  51582143


def main():
    total_equity_d1 = 0
    total_dayPL_d1 = 0
    total_grossPL_d1 = 0

    # server = 'real'
    # kind='dataframe'

    # print(f'{server} accounts:')

    # con = fxcmpy.fxcmpy(access_token=FXCM_ACCESS_TOKEN_REAL_1, server=server, log_level='error')
    # accounts_summary = con.get_accounts_summary(kind=kind)
    # print(accounts_summary[['balance', 'equity', 'dayPL', 'grossPL']])
    # con.close()

    # con = fxcmpy.fxcmpy(access_token=FXCM_ACCESS_TOKEN_REAL_2, server=server, log_level='error')
    # accounts_summary = con.get_accounts_summary(kind=kind)
    # print(accounts_summary[['balance', 'equity', 'dayPL', 'grossPL']])
    # con.close()

    server = 'demo'
    kind = 'dataframe'

    print(f'{server} accounts:')

    con = fxcmpy.fxcmpy(access_token=FXCM_ACCESS_TOKEN_DEMO_1, server=server, log_level='error')
    accounts_summary = con.get_accounts_summary(kind=kind)
    print(accounts_summary[['balance', 'equity', 'dayPL', 'grossPL']])
    con.close()
    time.sleep(20)

    con = fxcmpy.fxcmpy(access_token=FXCM_ACCESS_TOKEN_DEMO_2, server=server, log_level='error')
    accounts_summary = con.get_accounts_summary(kind=kind)
    print(accounts_summary[['balance', 'equity', 'dayPL', 'grossPL']])
    con.close()
    time.sleep(20)

    con = fxcmpy.fxcmpy(access_token=FXCM_ACCESS_TOKEN_DEMO_3, server=server, log_level='error')
    accounts_summary = con.get_accounts_summary(kind=kind)
    print(accounts_summary[['balance', 'equity', 'dayPL', 'grossPL']])
    con.close()
    time.sleep(20)

    con = fxcmpy.fxcmpy(access_token=FXCM_ACCESS_TOKEN_DEMO_4, server=server, log_level='error')
    accounts_summary = con.get_accounts_summary(kind=kind)
    print(accounts_summary[['balance', 'equity', 'dayPL', 'grossPL']])
    con.close()
    time.sleep(20)

    con = fxcmpy.fxcmpy(access_token=FXCM_ACCESS_TOKEN_DEMO_5, server=server, log_level='error')
    accounts_summary = con.get_accounts_summary(kind=kind)
    print(accounts_summary[['balance', 'equity', 'dayPL', 'grossPL']])
    con.close()
    time.sleep(20)

    con = fxcmpy.fxcmpy(access_token=FXCM_ACCESS_TOKEN_DEMO_6, server=server, log_level='error')
    accounts_summary = con.get_accounts_summary(kind=kind)
    print(accounts_summary[['balance', 'equity', 'dayPL', 'grossPL']])
    con.close()
    time.sleep(20)

    con = fxcmpy.fxcmpy(access_token=FXCM_ACCESS_TOKEN_DEMO_7, server=server, log_level='error')
    accounts_summary = con.get_accounts_summary(kind=kind)
    print(accounts_summary[['balance', 'equity', 'dayPL', 'grossPL']])
    con.close()
    time.sleep(20)

    con = fxcmpy.fxcmpy(access_token=FXCM_ACCESS_TOKEN_DEMO_8, server=server, log_level='error')
    accounts_summary = con.get_accounts_summary(kind=kind)
    print(accounts_summary[['balance', 'equity', 'dayPL', 'grossPL']])
    con.close()
    time.sleep(20)

    con = fxcmpy.fxcmpy(access_token=FXCM_ACCESS_TOKEN_DEMO_9, server=server, log_level='error')
    accounts_summary = con.get_accounts_summary(kind=kind)
    print(accounts_summary[['balance', 'equity', 'dayPL', 'grossPL']])
    con.close()
    time.sleep(20)

    con = fxcmpy.fxcmpy(access_token=FXCM_ACCESS_TOKEN_DEMO_10, server=server, log_level='error')
    accounts_summary = con.get_accounts_summary(kind=kind)
    print(accounts_summary[['balance', 'equity', 'dayPL', 'grossPL']])
    con.close()


if __name__ == "__main__":
    main()
