from gym.spaces import Box
from dm_control.suite.wrappers import pixels
from dm_control.rl import environment
import numpy as np


class PixelsEnv(environment.Base):

    def __init__(self, env):
        self._env = pixels.Wrapper(pixels.Wrapper(env), pixels_only=True)
        action_spec = self._env.action_spec()
        time_step = self._env.reset()
        observation_dm = time_step.observation["pixels"]
        screen_height_dm = observation_dm.shape[0]
        screen_width_dm = observation_dm.shape[1]
        screen_depth_dm = observation_dm.shape[2]
        self.observation_space = Box(low=0, high=255, shape=(screen_height_dm, screen_width_dm, screen_depth_dm),
                                     dtype=np.uint8)
        self.action_space = Box(-1., 1., shape=action_spec.shape, dtype='float32')
        self.random_action = np.random.uniform(action_spec.minimum,
                                               action_spec.maximum,
                                               size=action_spec.shape)

    def action_spec(self):
        return self.action_space

    def reset(self):
        time_step = self._env.reset()
        return time_step.observation["pixels"]

    def step(self, action):
        time_step = self._env.step(action)
        return time_step.observation["pixels"], time_step.reward, time_step.last(), None

    def observation_spec(self):
        return self.observation_space

    def step_spec(self):
        return self._env.step_spec()

