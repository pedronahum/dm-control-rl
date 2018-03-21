

def arg_parser():
    """
        Create an empty argparse.ArgumentParser.
        """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def dm_control_parser():
    """
        Create an argparse.ArgumentParser for DM Control.
        """
    parser = arg_parser()
    parser.add_argument('--domain_name', help='Domain Name', type=str, default='cartpole')
    parser.add_argument('--task_name', help='Task Name', type=str, default='swingup')
    parser.add_argument('--num_timesteps', type=int, default=int(10e4))
    return parser
