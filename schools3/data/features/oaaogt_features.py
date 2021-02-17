import sqlalchemy as sql
from sqlalchemy.types import VARCHAR

from schools3.config.data import db_tables
from schools3.config.data.features import oaaogt_features_config
from schools3.data.features.features_table import FeaturesTable
from schools3.data.features.processors.composite_feature_processor import CompositeFeatureProcessor
from schools3.data.features.processors.categorical_feature_processor import CategoricalFeatureProcessor
from schools3.data.features.processors.oaaogt_processor import OAAOGTProcessor
from schools3.data.features.processors.impute_null_processor import ImputeNullProcessor, ImputeBy

# table of features related to OAA/OGT test scores. Derived from `clean.oaaogt`
class OAAOGTFeatures(FeaturesTable):
    def __init__(self):
        feature_cols = [
            sql.Column('sixth_read_pl', VARCHAR),
            sql.Column('sixth_math_pl', VARCHAR),
            sql.Column('sixth_write_pl', VARCHAR),
            sql.Column('sixth_ctz_pl', VARCHAR),
            sql.Column('sixth_science_pl', VARCHAR),

            sql.Column('seventh_read_pl', VARCHAR),
            sql.Column('seventh_math_pl', VARCHAR),
            sql.Column('seventh_write_pl', VARCHAR),

            sql.Column('eighth_read_pl', VARCHAR),
            sql.Column('eighth_math_pl', VARCHAR),
            sql.Column('eighth_science_pl', VARCHAR),
            sql.Column('eighth_socstudies_pl', VARCHAR),
        ]

        pre_features_processor = CompositeFeatureProcessor([
            # OAAOGTProcessor(), # adding this reduces testing perf
            CategoricalFeatureProcessor(
                column_list=oaaogt_features_config.categorical_columns
            )
        ])

        post_features_processor = CompositeFeatureProcessor([
            ImputeNullProcessor(fill_unspecified=ImputeBy.MEAN)
        ])

        super(OAAOGTFeatures, self).__init__(
            table_name='oaaogt_features',
            feature_cols=feature_cols,
            categorical_cols=[],
            post_features_processor=post_features_processor,
            pre_features_processor=pre_features_processor,
            high_school_features=False,
        )

    def get_data_query(self):
        oaaogt = db_tables.clean_oaaogt_table

        return sql.select([
            oaaogt.c.student_lookup,
            oaaogt.c.sixth_read_pl,
            oaaogt.c.sixth_math_pl,
            oaaogt.c.sixth_write_pl,
            oaaogt.c.sixth_ctz_pl,
            oaaogt.c.sixth_science_pl,
            oaaogt.c.seventh_read_pl,
            oaaogt.c.seventh_math_pl,
            oaaogt.c.seventh_write_pl,
            oaaogt.c.eighth_read_pl,
            oaaogt.c.eighth_math_pl,
            oaaogt.c.eighth_science_pl,
            oaaogt.c.eighth_socstudies_pl,
        ]).\
        distinct(oaaogt.c.student_lookup)
