import gym
from d3rlpy.algos import SAC
from d3rlpy.envs import AsyncBatchEnv

if __name__ == '__main__':  # this is necessary if you use AsyncBatchEnv
    env = AsyncBatchEnv(
        [lambda: gym.make('Hopper-v2') for _ in range(5)])  # distributing 5 environments in different processes

    sac = SAC(use_gpu=True)
    sac.fit_batch_online(env)  # train with 5 environments concurrently
