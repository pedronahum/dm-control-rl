from common.cmd_util import dm_control_parser
from dm_control import suite
from dm_control.suite.wrappers import pixels
from moviepy.editor import ImageSequenceClip
import numpy as np


def main(args):

    # Load one task:
    env = suite.load(domain_name=args.domain_name, task_name=args.task_name)

    # Wrap the environment to obtain the pixels
    env = pixels.Wrapper(env, pixels_only=False)

    # Step through an episode and print out reward, discount and observation.
    action_spec = env.action_spec()
    time_step = env.reset()
    observation_matrix = []

    while not time_step.last():
        action = np.random.uniform(action_spec.minimum,
                                   action_spec.maximum,
                                   size=action_spec.shape)
        time_step = env.step(action)
        observation_dm = time_step.observation["pixels"]
        observation_matrix.append(observation_dm)

    clip = ImageSequenceClip(observation_matrix, fps=50)
    clip.write_gif("img/quickstart-humanoid-run.gif")


if __name__ == '__main__':
    arg = dm_control_parser().parse_args()
    main(arg)
