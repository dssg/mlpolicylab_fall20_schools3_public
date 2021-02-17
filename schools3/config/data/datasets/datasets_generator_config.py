import sys
import sqlalchemy as sql
import pandas as pd

from schools3.config import base_config
from schools3.config.data import db_config, db_tables


config = base_config.Config()

def get_years_for_grade(grade):
    engine = sql.create_engine(db_config.engine_url)
    all_snapshots = db_tables.clean_all_snapshots_table

    query = sql.select([
        sql.distinct(all_snapshots.c.school_year)
    ]).where(
        all_snapshots.c.grade == grade
    ).order_by(
        all_snapshots.c.school_year
    )

    return pd.read_sql(query, engine).school_year.to_list()


config.label_windows = {
    9: 3,
    10: 2,
    11: 1
}

config.update_frequency = 1

config.years_per_grade = {k: get_years_for_grade(k) for k in range(9, 12)}

sys.modules[__name__] = config
