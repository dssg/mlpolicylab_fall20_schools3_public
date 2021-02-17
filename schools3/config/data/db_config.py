import sys
from schools3.config import base_config

config = base_config.Config()

# connection config
config.dialect = 'postgresql'
config.driver = 'psycopg2'
config.host = 'YOUR-HOST'
config.user = 'YOUR-USER'
config.db_name = 'YOUR-DB'

config.engine_url = \
    f'{config.dialect}+{config.driver}://{config.user}@{config.host}/{config.db_name}'


# database particulars
config.read_schema_name = 'clean'
config.write_schema_name = 'sketch'

config.other_schemas = ['public', 'acs']

sys.modules[__name__] = config
