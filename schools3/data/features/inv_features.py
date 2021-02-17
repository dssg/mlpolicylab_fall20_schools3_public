import sqlalchemy as sql
import inflection
from sqlalchemy.sql import func as db_func
from schools3.config.data import db_tables
from schools3.config.data.features import inv_features_config, features_config
from schools3.data.features.pivot_block_features import PivotBlockFeatures
from schools3.data.features.processors.composite_feature_processor import CompositeFeatureProcessor
from schools3.data.features.processors.pivot_processor import PivotProcessor
from schools3.data.features.processors.impute_null_processor import ImputeNullProcessor


class InvFeatures(PivotBlockFeatures):
    def __init__(self):
        all_inv = db_tables.clean_intervention_table
        index_cols_dict = {
            'student_lookup': all_inv.c.student_lookup,
            'school_year': sql.cast(
                                db_func.substr(
                                    all_inv.c.school_year,
                                    db_func.length(all_inv.c.school_year) - 3, 4
                                ),
                                sql.INT
                        ).label('school_year'),
            'grade': all_inv.c.grade
        }

        super(InvFeatures, self).__init__(
            table_name=inflection.underscore(InvFeatures.__name__),
            categorical_cols=inv_features_config.categorical_columns,
            post_features_processor=CompositeFeatureProcessor([
                ImputeNullProcessor(fill_unspecified=0)
            ]),
            data_table=all_inv,
            blocking_col=all_inv.c.inv_group,
            index_cols_dict=index_cols_dict
        )
