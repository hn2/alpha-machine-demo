import gym


def register(id, entry_point, **kwargs):
    env_specs = gym.envs.registry.env_specs

    if id in env_specs.keys():
        del env_specs[id]

    gym.register(
        id=id,
        entry_point=entry_point,
        **kwargs
    )


# Register modified versions of existing environments
register(id='PfFxEnv-v0',
         entry_point='gym_pf_fx.envs:PfFxEnv',
         kwargs=dict(data=None,
                     account_currency=None,
                     instruments=None,
                     lookback=10,
                     random_episode_start=True,
                     cash=1e3,
                     max_slippage_percent=1e-2,
                     lot_size='Micro',
                     leverage=2e1,
                     pip_size=[0.0001],
                     pip_spread=[3],
                     compute_position='long_and_short',  # long_only, short_only, long_and_short
                     compute_indicators='all',
                     compute_reward='log_returns',  # returns log_returns
                     meta_rl=False,
                     verbose=False)
         )
