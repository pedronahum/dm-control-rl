from mpi4py import MPI
from common.cmd_util import dm_control_parser
from common.pixels_env import PixelsEnv
from common.basic_env import BasicEnv
from dm_control import suite
from baselines import logger
from dm_control_baselines.ppo1.mlp_policy import MlpPolicy
from dm_control_baselines.trpo_mpi import trpo_mpi


def train(args, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    env = suite.load(domain_name=args.domain_name, task_name=args.task_name)

    # TODO: Implement a way to add seeds
    # env = env.random(workerseed)

    if args.use_pixels:
        env = PixelsEnv(env)
    else:
        env = BasicEnv(env)

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_size=32, num_hid_layers=2)

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                   max_timesteps=args.num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)

    env.close()


def main():
    args = dm_control_parser().parse_args()

    logger.configure(dir='/tmp', format_strs=['tensorboard', 'csv'])

    train(args, seed=args.seed)


if __name__ == '__main__':
    main()
