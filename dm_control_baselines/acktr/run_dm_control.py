
from common.cmd_util import dm_control_parser
from common.pixels_env import PixelsEnv
from common.basic_env import BasicEnv
from dm_control import suite
from baselines import logger

from dm_control_baselines.acktr.acktr_cont import learn
from dm_control_baselines.acktr.policies import GaussianMlpPolicy
from dm_control_baselines.acktr.value_functions import NeuralNetValueFunction
import tensorflow as tf


def train(args, env):

    with tf.Session(config=tf.ConfigProto()):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        learn(env, policy=policy, vf=vf,
              gamma=0.99, lam=0.97, timesteps_per_batch=2500,
              desired_kl=0.002,
              num_timesteps=args.num_timesteps, animate=False)

        env.close()


def main():
    args = dm_control_parser().parse_args()

    logger.configure()

    env = suite.load(domain_name=args.domain_name, task_name=args.task_name)

    if args.use_pixels:
        env = PixelsEnv(env)
    else:
        env = BasicEnv(env)
    train(args, env)


if __name__ == '__main__':
    main()
