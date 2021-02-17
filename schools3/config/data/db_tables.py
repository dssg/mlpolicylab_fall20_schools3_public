import sys
import sqlalchemy as sql
from schools3.config import base_config
from schools3.config.data import db_config


config = base_config.Config()
engine = sql.create_engine(db_config.engine_url)


def add_tables(schema_name):
    '''
        Adds tables into config with name = {schema_name}_{table_name}_table
    '''
    meta = sql.MetaData(
                bind=engine,
                schema=schema_name
            )

    meta.reflect()

    for name in meta.tables:
        config_table_name = name.replace('.', '_') + '_table'
        if config_table_name not in dir(config):
            setattr(config, config_table_name, meta.tables[name])


def reload_tables():
    add_tables(db_config.read_schema_name)
    add_tables(db_config.write_schema_name)
    for s in db_config.other_schemas:
        add_tables(s)

reload_tables()
config.reload_tables = reload_tables

sys.modules[__name__] = config
