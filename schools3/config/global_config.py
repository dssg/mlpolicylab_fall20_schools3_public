import sys
import os
from pathlib import Path
import inspect
from getpass import getuser
from datetime import datetime
from schools3.config import base_config
import tensorflow as tf

config = base_config.Config()

config.save_dir = os.path.join(
                    str(Path(__file__).parent.parent.absolute()), 'gen/'
                )

config.debug_dir = os.path.join(
                    str(Path(__file__).parent.parent.absolute()), 'ml/debug/'
                )

config.etc_invfeat_dir = os.path.join(
                    str(Path(__file__).parent.parent.absolute()), 'config/data/etc/'
                )


if not os.path.exists(config.save_dir):
    os.mkdir(config.save_dir)


def join_path(path, folder, *kwargs):
    joined_path = os.path.join(path, folder,*kwargs)

    if not os.path.exists(joined_path):
        os.makedirs(joined_path)

    return joined_path

def join_path_file(path, folder):
    joined_file = os.path.join(path, folder)

    return joined_file


def get_save_path(file_name, use_user_time=False):
    caller_file = str(Path(inspect.stack()[1].filename).parent)
    base = str(Path(__file__).parent.parent.absolute())
    assert caller_file.startswith(base), 'get_cur_save_dir only works when called inside of schools3/'

    save_dir = os.path.join(config.save_dir, caller_file[len(base):].strip('/'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if use_user_time:
        base, ext = os.path.splitext(file_name)
        user = getuser()
        now = datetime.now().strftime('%b_%d__%H_%M')
        file_name = '_'.join([base, user, now]) + ext

    path = os.path.join(save_dir, file_name)
    base_dir = os.path.dirname(path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    return path

config.join_path = join_path
config.join_path_file = join_path_file
config.get_save_path = get_save_path

config.num_threads = 4

tf.config.threading.set_intra_op_parallelism_threads(config.num_threads)
tf.config.threading.set_inter_op_parallelism_threads(config.num_threads)

sys.modules[__name__] = config
