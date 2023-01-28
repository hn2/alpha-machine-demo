import pandas as pd
from sqlalchemy import create_engine

#   Which Reinforcement learning-RL algorithm to use where, when and in what scenario? https://medium.com/datadriveninvestor/which-reinforcement-learning-rl-algorithm-to-use-where-when-and-in-what-scenario-e3e7617fb0b1
#   cd util_lib
#   python -m tensorboard.main --logdir=.
#   tensorboard --port 7000 --logdir=.
#   tensorboard --logdir=.

DELIMITER = '-'


def main():
    alchemyEngine = create_engine('postgresql+psycopg2://postgres:dD33dD33@localhost:5432', pool_recycle=3600);

    dbConnection = alchemyEngine.connect();

    sql_text = 'set AUTOCOMMIT on; drop database optuna; create database optuna;'

    dbConnection.execute(sql_text)

    pd.set_option('display.expand_frame_repr', False);

    dbConnection.close();


if __name__ == "__main__":
    main()
