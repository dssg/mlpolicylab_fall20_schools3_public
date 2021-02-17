import sys
from schools3.config import base_config

config = base_config.Config()

config.pl_score = {
    'failed': 0,
    'limited': 0,
    'below basic': 0,

    'basic': 1,
    'basi': 1,

    'proficient': 2,
    'passed': 2,

    'accelerated': 3,

    'advanced': 4
}

config.ordinal_post_fix = '_ordinal'

sys.modules[__name__] = config
