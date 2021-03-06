
from common.cmd_util import dm_control_parser
from common.pixels_env import PixelsEnv
from common.basic_env import BasicEnv
from dm_control import suite
from baselines import logger


from baselines.common import tf_util as U


def train(args, env):
    from dm_control_baselines.ppo1 import pposgd_simple, cnn_policy, mlp_policy
    U.make_session(num_cpu=1).__enter__()

    if args.use_pixels:

        def policy_fn(name, ob_space, ac_space):
            return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    kind='small')
        pposgd_simple.learn(env, policy_fn,
                            max_timesteps=args.num_timesteps,
                            timesteps_per_actorbatch=2048,
                            clip_param=0.2, entcoeff=0.0,
                            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                            gamma=0.99, lam=0.95, schedule='linear',
                            )
    else:
        def policy_fn(name, ob_space, ac_space):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                        hid_size=64, num_hid_layers=2)

        pposgd_simple.learn(env, policy_fn,
                            max_timesteps=args.num_timesteps,
                            timesteps_per_actorbatch=2048,
                            clip_param=0.2, entcoeff=0.0,
                            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                            gamma=0.99, lam=0.95, schedule='linear',
                            )
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
