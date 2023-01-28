import datetime
import json
import pickle
import uuid
from os.path import join as path_joine

from utils_lib.my_globals import OPTUNA_DIR

from best_model_params import get_from_db

DELIMITER = '-'
PRINT_TO_FILE = False
NUMBER_OF_INSTRUMENTS = 7


def main():
    #   for online_algorithm in ['sac','td3']:
    for online_algorithm in ['ppo']:
        for market in ['fxcm']:
            #   for market in ['oanda', 'fxcm']:
            #   for resolution in ['day', 'hour']:
            for resolution in ['day']:

                now = datetime.datetime.now()

                #   study_name = '\'fx-' + str(NUMBER_OF_INSTRUMENTS) + DELIMITER + online_algorithm + DELIMITER + market + DELIMITER + resolution + '\''

                #   study_name = '\'fx-' + str(NUMBER_OF_INSTRUMENTS) + DELIMITER + online_algorithm.lower() + DELIMITER + market + DELIMITER + resolution + DELIMITER + str(now.day) + DELIMITER + str(now.month) + DELIMITER + str(now.year) + '\''

                #   study_name = "'fx-7-td3-log_returns-oanda-day-14-2-2021'"

                #   study_name = "'fx-7-td3-log_returns-oanda-day-12-2-2021'"

                #   study_name = "'fx-7-td3-log_returns-oanda-day-13-2-2021'"

                #   study_name = "'fx-7-td3-log_returns-oanda-day-14-2-2021'"

                study_name = "'fx-ppo-log_returns-fxcm-day-1-6-2021'"

                df, best_params = get_from_db(study_name)

                print('\n')
                print("------------------------------------------------------")
                print(f'Online Algorithm: {online_algorithm} Market: {market}, Resolution: {resolution}')
                print("------------------------------------------------------")
                print('\n')
                print(df)
                print('\n')
                print(best_params)

                if PRINT_TO_FILE:
                    print(path_joine(OPTUNA_DIR, study_name[1:-1] + ".json"))

                    path = path_joine(OPTUNA_DIR, 'best-params')
                    v_uuid = (uuid.uuid4().hex)[:8]

                    js = json.dumps(best_params)

                    with open(path_joine(path, study_name[1:-1] + "_" + v_uuid + ".json"), "w") as f:
                        f.write(js)

                    with open(path_joine(path, study_name[1:-1] + "_" + v_uuid + ".pkl"), "wb") as f:
                        pickle.dump(js, f)


if __name__ == "__main__":
    main()
