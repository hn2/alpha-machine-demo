import os
import os.path
from os.path import join as path_join

#   models_dir = r'C:\alpha-machine\src\drl-forex\models'
#   models_dir = r"E:\alpha-machine\models\sb3\day\all"

# models_dir = r"C:\alpha-machine\models\forex\oanda\daily\prod"
models_dir = r"C:\alpha-machine\models\forex\oanda\daily\prod"


def main():
    sub_folders_old = [name for name in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, name))]
    print(sub_folders_old)

    sub_folders_new = [
        # name.replace('with_', 'w_').replace('train_', '').replace('leverage_', 'lev_').replace('callback',
        #                                                                                        'cb').replace(
        #     'random_episode_start', 'res') for name in sub_folders_old]

        name.replace('300', '332') for name in sub_folders_old]

    print(sub_folders_new)

    for folder_name_old, folder_name_new in zip(sub_folders_old, sub_folders_new):
        print(f'Renaming {folder_name_old} to {folder_name_new}')

        os.rename(path_join(models_dir, folder_name_old), path_join(models_dir, folder_name_new))


if __name__ == "__main__":
    main()
