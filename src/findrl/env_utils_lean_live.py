import sys

sys.path.append("..")
from envs.gym_pf_fx.envs import PfFxEnv


def create_env(data,
               instruments,
               lookback,
               random_episode_start,
               cash,
               max_slippage_percent,
               lot_size,
               leverage,
               pip_size,
               pip_spread,
               compute_position,
               compute_indicators,
               compute_reward,
               verbose):
    v_env = PfFxEnv(data=data,
                    instruments=instruments,
                    lookback=lookback,
                    random_episode_start=random_episode_start,
                    cash=cash,
                    max_slippage_percent=max_slippage_percent,
                    lot_size=lot_size,
                    leverage=leverage,
                    pip_size=pip_size,
                    pip_spread=pip_spread,
                    compute_position=compute_position,
                    compute_indicators=compute_indicators,
                    compute_reward=compute_reward,
                    verbose=verbose)

    return v_env
