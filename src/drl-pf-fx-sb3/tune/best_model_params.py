import json

import pandas as pd
from sqlalchemy import create_engine

#   Which Reinforcement learning-RL algorithm to use where, when and in what scenario? https://medium.com/datadriveninvestor/which-reinforcement-learning-rl-algorithm-to-use-where-when-and-in-what-scenario-e3e7617fb0b1
#   cd util_lib
#   python -m tensorboard.main --logdir=.
#   tensorboard --port 7000 --logdir=.
#   tensorboard --logdir=.

DELIMITER = '-'


def get_from_db(study_name):
    '''
    storage = optuna.storages.RDBStorage(
        #   url='postgresql://hannan:dD33dD33@optuna.cu3liabuijge.us-east-1.rds.amazonaws.com:5432/optuna', #   aws
        url='postgresql://postgres:dD33dD33@localhost:5432/optuna', #   local
        engine_kwargs={
            'pool_size': 20,
            'max_overflow': 0
        }
    )
    '''

    #   alchemyEngine   = create_engine('postgresql+psycopg2://postgres:dD33dD33@localhost:5432/optuna', pool_recycle=3600)

    alchemyEngine = create_engine('mysql+mysqlconnector://hannan:dD33dD33@localhost:3306/optuna', pool_recycle=3600)

    dbConnection = alchemyEngine.connect()

    sql_text = f"""
                select tp.trial_id, tv.value , tp.param_name, tp.param_value, tp.distribution_json 
                from studies s, trials t, trial_values tv, trial_params tp 
                where s.study_name = {study_name}
                and t.state = 'COMPLETE'
                and t.study_id = s.study_id 
                and t.trial_id = tv.trial_id 
                and t.trial_id = tp.trial_id 
                order by tv.value desc
                limit 13
            """

    df = pd.read_sql(sql_text, dbConnection)

    pd.set_option('display.expand_frame_repr', False)

    dbConnection.close()

    #   Convert to dict

    params = {}

    for index, row in df.iterrows():

        param_name = row['param_name']
        param_value = row['param_value']
        distribution_json = row['distribution_json']

        distributions = json.loads(str(distribution_json))

        if distributions['name'] == 'CategoricalDistribution':
            param_value = int(row['param_value'])
            #   print(param_value)
            categories = distributions['attributes']['choices']
            #   print(categories)
            param_value = categories[param_value]

        params[param_name] = param_value

    return df, params
