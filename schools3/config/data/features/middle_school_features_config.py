import sys
from schools3.config import base_config
from schools3.config.data.features import features_config

config = base_config.Config()

config.start_gpa_grade = features_config.min_grade
config.end_gpa_grade = 8

sys.modules[__name__] = config
