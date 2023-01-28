#   sudo python3 /home/ubuntu/alpha-machine/src/drl-forex/test-cron-python.py
from os.path import join as path_join


def main():
    path = r'/home/ubuntu/alpha-machine/src/drl-forex/tests'
    #   path = r'C:\alpha-machine\src\drl-fx\tests'
    line = 'cron works python'
    with open(path_join(path, 'readme.txt'), 'a+') as f:
        f.writelines(line + '\n')


if __name__ == "__main__":
    main()
