import tensorflow as tf
import re
import os


def define_saver(exclude=None):
    """Create a saver for the variables we want to checkpoint.

  Args:
    exclude: List of regexes to match variable names to exclude.

  Returns:
    Saver object.
  """
    variables = []
    exclude = exclude or []
    exclude = [re.compile(regex) for regex in exclude]
    for variable in tf.global_variables():
        if any(regex.match(variable.name) for regex in exclude):
            continue
        variables.append(variable)
    saver = tf.train.Saver(variables, keep_checkpoint_every_n_hours=5)
    return saver


def get_session():
    """Returns recently made Tensorflow session"""
    return tf.get_default_session()


def load_state(fname):
    saver = tf.train.Saver()
    saver.restore(get_session(), fname)


def save_state(fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(get_session(), fname)
