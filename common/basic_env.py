from gym.spaces import Box
from dm_control.rl import environment
import numpy as np
from numpy import array


def process_observation(observation):
    ob_space = []
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            for i in value:
                ob_space.append(i)
        else:
            ob_space.append(value)
    return array(ob_space)


class BasicEnv(environment.Base):

    def __init__(self, env):
        self._env = env
        action_spec = self._env.action_spec()
        time_step = self._env.reset()
        observation_dm = time_step.observation
        ob_space = process_observation(observation_dm)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=ob_space.shape,
                                     dtype=np.uint8)
        self.action_space = Box(action_spec.minimum, action_spec.maximum, dtype='float32')
        self.random_action = np.random.uniform(action_spec.minimum,
                                               action_spec.maximum,
                                               size=action_spec.shape)

    def action_spec(self):
        return self.action_space

    def reset(self):
        time_step = self._env.reset()
        return process_observation(time_step.observation)

    def step(self, action):
        time_step = self._env.step(action)
        return process_observation(time_step.observation), time_step.reward, time_step.last(), None

    def observation_spec(self):
        return self.observation_space

    def step_spec(self):
        return self._env.step_spec()
