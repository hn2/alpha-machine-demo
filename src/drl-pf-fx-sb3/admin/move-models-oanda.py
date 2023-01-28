import os
import os.path
import shutil
from os.path import join as path_join

#   models_dir = r'C:\alpha-machine\src\drl-forex\models'
#   models_dir = r"E:\alpha-machine\models\sb3\day\all"

source_dir = r"C:\alpha-machine\models\forex\oanda\daily\prod"
target_dir = r"C:\alpha-machine\models\forex\oanda\daily\moved"


def move_model(name):
    try:
        file_names = os.listdir(path_join(source_dir, name))
        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name[:-1]), path_join(target_dir, file_name[:-1]))
    except Exception as e:
        print(e)


def main():
    with open(r'move-oanda.txt') as file:
        lines = file.readlines()
        for line in lines:
            if line == '\n':
                continue
            print(line)
            move_model(line)


if __name__ == "__main__":
    main()
