import fileinput
import json
import os
import pickle
import sys
from datetime import datetime, timedelta
from os.path import join as path_join


def replace_in_file(file_name, search_for, replace_with):
    with fileinput.input(file_name, inplace=True) as file:
        for line in file:
            if line.strip().startswith(search_for):
                line = replace_with
            sys.stdout.write(line)


def _ts_to_dt(ts):
    return datetime.fromtimestamp(ts)


def get_subfolders(models_dir, files_lookback_hours, include_patterns):
    subfolders = [f.path for f in os.scandir(models_dir) if
                  f.is_dir() and _ts_to_dt(f.stat().st_ctime) > (datetime.now() - timedelta(hours=files_lookback_hours))
                  # and any(str(f) in string for string in INCLUDE_PATTERNS)]
                  and [ele for ele in include_patterns if (ele in str(f))]]

    return subfolders


def get_timestamps(models_dir, files_lookback_hours, include_patterns):
    timestamps = [_ts_to_dt(f.stat().st_ctime) for f in os.scandir(models_dir) if
                  f.is_dir() and _ts_to_dt(f.stat().st_ctime) > (datetime.now() - timedelta(hours=files_lookback_hours))
                  # and any(str(f) in string for string in INCLUDE_PATTERNS)]
                  and [ele for ele in include_patterns if (ele in str(f))]]

    return timestamps


def get_models(models_dir, files_lookback_hours, include_patterns):
    v_subfolders = get_subfolders(models_dir, files_lookback_hours, include_patterns)
    v_models = [x.split('\\')[-1] for x in v_subfolders]

    return v_models


def get_subdir(path):
    for sd in os.scandir(path):
        if sd.is_dir():
            subdir = sd.path

    return subdir


def get_subdir_with_pattern(path, pattern):
    for sd in os.scandir(path):
        if sd.is_dir() and pattern in str(sd.path):
            subdir = sd.path

    return subdir


def load_dict_from_json(json_file_name):
    with open(json_file_name) as json_file:
        data = json.load(json_file)

    # with open(json_file_name) as json_file:
    #     json_file_content = json_file.read()
    #
    # a_dict = json.loads(json_file_content)

    return data


def dump_objects_to_pickle(file_name, list_of_objects):
    with open(file_name, "wb") as f:
        pickle.dump(len(list_of_objects), f)
        for object in list_of_objects:
            pickle.dump(object, f)


def load_objects_from_pickle(file_name):
    list_of_objects = []
    with open(file_name, "rb") as f:
        for _ in range(pickle.load(f)):
            list_of_objects.append(pickle.load(f))

    return list_of_objects


def dump_dict_to_pickle(file_name, dict_object):
    with open(file_name, "wb") as f:
        pickle.dump(dict_object, f)


def load_dict_from_pickle(file_name):
    with open(file_name, "rb") as f:
        dict_object = pickle.load(f)

    return dict_object


def get_instruments_in_model(models_dir, model_name):
    env_attributes = load_dict_from_pickle(path_join(*[models_dir, model_name, 'online', 'env.pkl']))
    instruments = env_attributes.instruments

    return instruments
