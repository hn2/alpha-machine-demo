import os
import shutil
from datetime import datetime, timedelta
from os.path import join as path_join

#   models_dir = r'C:\alpha-machine\src\drl-forex\models'
MODELS_DIR = r"E:\alpha-machine\models\sb3"
OFFLINE_ALGORITHMS = ['BCQ', 'BEAR', 'AWR', 'CQL', 'AWAC', 'CRR', 'PLAS', 'MOPO', 'COMBO']
RESOLUTION = 'daily'  # day   hour   old
SUBDIR = 'all'
DELTA_FILES = 1000
INCLUDE_PATTERNS = ['2000']


def ts_to_dt(ts):
    return datetime.fromtimestamp(ts)


def get_subdir(path):
    for sd in os.scandir(path):
        if sd.is_dir():
            subdir = sd.path

    return subdir


def main():
    v_models_dir = path_join(MODELS_DIR, RESOLUTION)
    v_models_dir = path_join(v_models_dir, SUBDIR)
    print(f'MODELS_DIR: {v_models_dir}')

    subfolders = [f.path for f in os.scandir(v_models_dir) if
                  f.is_dir() and ts_to_dt(f.stat().st_atime) > (datetime.now() - timedelta(days=DELTA_FILES))
                  # and any(str(f) in string for string in INCLUDE_PATTERNS)]
                  and [ele for ele in INCLUDE_PATTERNS if (ele in str(f))]]

    for v_model_dir in subfolders:

        v_offline_models_dir = path_join(v_model_dir, 'offline')

        try:
            print(f'Removing directory {v_offline_models_dir}')
            shutil.rmtree(v_offline_models_dir)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
