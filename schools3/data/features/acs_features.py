import pandas as pd
import requests
import sqlalchemy as sql
import sqlalchemy.sql.functions as db_func
import sqlalchemy.sql.expression as db_expr
from sqlalchemy.orm import aliased
from sqlalchemy.types import ARRAY, INT, VARCHAR, FLOAT
from schools3.config.data import db_tables
from schools3.config.data import db_config
from schools3.config.data.features import features_config
import schools3.config.data.features.acs_features_config as config
from schools3.data.features.features_table import FeaturesTable
from schools3.data.features.processors.composite_feature_processor import CompositeFeatureProcessor
from schools3.data.features.processors.acs_feature_processor import ACSFeatureProcessor
from schools3.data.acs.acs_table import ACSTable

# table of features derived from ACS data
class ACSFeatures(FeaturesTable):
    def __init__(self):
        group_vars = config.get_group_variables()
        acs_feats = list(pd.core.common.flatten([v.keys() for k, v in group_vars.items()]))

        cols = [sql.Column(feat, FLOAT) for feat in acs_feats]

        pre_feature_processor = CompositeFeatureProcessor([
            ACSFeatureProcessor(),
        ])
        post_feature_processor = CompositeFeatureProcessor([])
        super(ACSFeatures, self).__init__(
            table_name='acs_features',
            feature_cols=cols,
            categorical_cols=config.categorical_columns,
            pre_features_processor=pre_feature_processor,
            post_features_processor=post_feature_processor,
        )

    def get_data_query(self):
        all_snapshots           = db_tables.clean_all_snapshots_table
        student_lookup          = all_snapshots.c.student_lookup
        school_year             = all_snapshots.c.school_year
        grade                   = all_snapshots.c.grade
        zipcode                 = all_snapshots.c.zip

        # get first 5 digits for zipcode
        processed_zipcode = sql.case([(zipcode == None, None)], else_=sql.func.substr(sql.cast(zipcode, VARCHAR), 1, 5))

        snapshots = sql.select([
                student_lookup,
                school_year,
                grade,
                processed_zipcode.label('zipcode'),
            ]).\
            distinct(student_lookup, school_year, grade).\
            where(
                student_lookup != None,
            ).\
            cte('acs_temp_a')

        to_join = [snapshots]
        for k, v in db_tables.__dict__.items():
            if k.startswith('acs_'):
                to_join.append(v)

        joined = to_join[0]
        for i in range(1, len(to_join)):
            if i == 1:
                on_clause = (joined.c.zipcode == to_join[i].c.zipcode)
            else:
                on_clause = (joined.c[to_join[0].name +'_zipcode'] == to_join[i].c.zipcode)

            joined = sql.join(
                        left=joined, right=to_join[i],
                        onclause=on_clause,
                        isouter=True
                    )

        cols = [c for c in joined.c if c.name != 'zipcode']        
        final = sql.select(cols).select_from(joined)

        return final

    def _get_df(self):
        # ensure that ACS data is downloaded
        engine = sql.create_engine(db_config.engine_url)
        acs_meta = sql.MetaData(bind=engine, schema=config.schema_name)
        acs_meta.reflect()
        acs_tables = [table.name for table in acs_meta.sorted_tables]

        group_variables = config.get_group_variables()
        
        for group_name, cols in group_variables.items():
            group_name = group_name.lower()
            if group_name not in acs_tables:
                t = ACSTable(group_name, cols)
                t.insert_from_df()
        
        db_tables.reload_tables()
        return self.query(self.get_data_query())
