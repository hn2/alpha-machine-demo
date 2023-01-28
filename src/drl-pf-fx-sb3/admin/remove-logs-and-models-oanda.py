import os
import os.path
import shutil
from os.path import join as path_join

#   models_dir = r'C:\alpha-machine\src\drl-forex\models'
#   models_dir = r"E:\alpha-machine\models\sb3\day\all"

models_dir = r"E:\alpha-machine\models\forex\oanda\daily\sb3-train-old"
#   models_dir = r"E:\alpha-machine\models\forex\fxcm\daily\prod"
logs_dir = r"E:\alpha-machine\src\drl-pf-fx-sb3\logs"


def delete_model(name):
    try:
        shutil.rmtree(path_join(models_dir, name[:-1]))
    except Exception as e:
        print(e)


def delete_log(name):
    try:
        shutil.rmtree(path_join(logs_dir, name[:-1] + '_0'))
    except Exception as e:
        print(e)

    try:
        os.remove(path_join(logs_dir, name[:-1] + '.monitor.csv'))
    except Exception as e:
        print(e)


def main():
    with open(r'remove-oanda.txt') as file:
        lines = file.readlines()
        for line in lines:
            if line == '\n':
                continue
            print(line)
            delete_model(line)


if __name__ == "__main__":
    main()
