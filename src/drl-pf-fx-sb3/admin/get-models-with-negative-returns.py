import collections
import glob

import pandas as pd
from utils_lib.my_globals import STATS_DIR

remove_file = r'C:\alpha-machine\src\drl-forex\admin\remove.txt'
# print(STATS_DIR)

# set all .xls files in your folder to list
excel_files = glob.glob(STATS_DIR + '\\*.xlsx')
# print(excel_files)
negative_returns_list = []

# for loop to aquire all excel files in folder
for excel_file in excel_files:
    # print(excel_file)
    df = pd.read_excel(excel_file, index_col=0)
    #   df['filter_column'] = df['Net Profit'].str.strip('%').astype(float)
    df['filter_column'] = df['Sharpe Ratio'].astype(float)
    df_filter = df[df['filter_column'] < 5]
    #   print(df_filter)
    #   df_filter.reset_index(level=0, inplace=True)
    #   print(df_filter.index.tolist())
    negative_returns_list.extend(df_filter.index.tolist())

values = [k for k, v in collections.Counter(negative_returns_list).items() if v > 1]

with open(remove_file, 'w') as output:
    for row in values:
        output.write(str(row) + '\n')
