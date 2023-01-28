import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SaveToDiskOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param models_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, check_freq, save_freq, lookback, online_algorithm, model_file_name, model_replay_buffer=None,
                 model_stats=None, save_replay_buffer=True, verbose=False):
        super(SaveToDiskOnBestTrainingRewardCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.lookback = lookback

        '''
        self.models_dir = models_dir
        self.save_path = os.path.join(models_dir, 'best_model')
        '''

        self.online_algorithm = online_algorithm
        self.model_file_name = model_file_name
        self.model_replay_buffer = model_replay_buffer
        self.model_stats = model_stats
        self.save_replay_buffer = save_replay_buffer
        self.verbose = verbose

        self.rewards = []
        self.best_mean_reward = -np.inf
        self.best_mean_reward_step = 0
        self.best_model = None

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        self.rewards.append(self.training_env.get_original_reward())

        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            # Mean training reward over the last 100 episodes
            # mean_reward = np.mean(self.rewards[-100:])

            mean_reward = np.mean(self.rewards[-self.lookback:])

            if self.verbose:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.4f}, Best mean reward step: {} Last mean reward per episode: {:.4f}".format(
                    self.best_mean_reward, self.best_mean_reward_step,
                    mean_reward))
                print("Last reward = ", self.rewards[-1])
                print("Reward buffer length = ", len(self.rewards))

            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.best_mean_reward_step = self.n_calls
                self.best_model = self.model

        if self.n_calls % self.save_freq == 0:
            if self.verbose:
                print("Saving new best model to {}".format(self.model_file_name))

            self.model.save(self.model_file_name)
            self.training_env.save(self.model_stats)

            if self.save_replay_buffer:
                if self.online_algorithm == 'SAC' or self.online_algorithm == 'TD3' or self.online_algorithm == 'DDPG':
                    try:
                        self.model.save_replay_buffer(self.model_replay_buffer)
                    except Exception as e:
                        print(e)

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
