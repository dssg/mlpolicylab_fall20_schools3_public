import sys
from schools3.config import base_config

config = base_config.Config()

config.fill_values = {
    # 'ethnicity': '',
    # 'school_name': '',
    # 'disability': '',
    # 'limited_english': 'N',
    # 'discipline_incidents': 0,
    # 'school_name': '',
    # 'disability': '',
    # 'num_transfers': 0,
}

sys.modules[__name__] = config
