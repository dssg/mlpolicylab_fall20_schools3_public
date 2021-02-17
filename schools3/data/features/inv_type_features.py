import sqlalchemy as sql
import inflection
from sqlalchemy.types import VARCHAR
from sqlalchemy.sql import func as db_func
from schools3.config.data import db_tables
from schools3.config.data.features import inv_features_config, features_config
from schools3.data.features.features_table import FeaturesTable
from schools3.data.features.processors.composite_feature_processor import CompositeFeatureProcessor
from schools3.data.features.processors.pivot_processor import PivotProcessor
#from schools3.data.features.processors.impute_null_processor import ImputeNullProcessor
from schools3.data.features.processors.inv_type_processor import InvTypeProcessor

class InvTypeFeatures(FeaturesTable):
    def __init__(self):
        feature_cols = [
            sql.Column('pivot_inv_group', VARCHAR),
            sql.Column('pivot_class1', VARCHAR),
            sql.Column('pivot_class2', VARCHAR),
            sql.Column('pivot_class3', VARCHAR),
            sql.Column('description', VARCHAR),
        ]
        super(InvTypeFeatures, self).__init__(
            table_name=inflection.underscore(InvTypeFeatures.__name__),
            feature_cols=feature_cols,
            categorical_cols=inv_features_config.categorical_columns,
            post_features_processor=CompositeFeatureProcessor([]),
            pre_features_processor=InvTypeProcessor(
                    index=['student_lookup', 'school_year', 'grade'],
                    columns=['pivot_inv_group', 'description'],
                    values=['pivot_class1', 'pivot_class2', 'pivot_class3']
            )
        )

    def get_data_query(self):
        all_inv = db_tables.clean_intervention_table

        student_lookup  = all_inv.c.student_lookup
        school_year     = sql.cast(
                            db_func.substr(
                                all_inv.c.school_year,
                                db_func.length(all_inv.c.school_year) - 3, 4
                            ),
                            sql.INT
                        ).label('school_year')
        grade           = all_inv.c.grade
        inv_group       = all_inv.c.inv_group
        description     = all_inv.c.description

        # FIXME: Make end year go upto to the last year on record
        student_years = sql.select([
                            student_lookup,
                            school_year.label('end_year'),
                            grade,
                        ]).distinct(
                            student_lookup,
                            school_year,
                            grade
                        ).where(
                            grade >= 9
                        ).alias('student_years')

        student_invs = sql.select([
                            student_lookup,
                            school_year,
                            grade,
                            inv_group,
                            description
                        ]).where(
                            grade >= features_config.min_grade
                        ).alias('student_invs')

        joined = sql.join(
            left=student_invs,
            right=student_years,
            onclause=sql.and_(
                        student_invs.c.student_lookup == student_years.c.student_lookup,
                        student_invs.c.school_year <= student_years.c.end_year
                    )
        )


        rate_col = db_func.count() * 1.0 / db_func.count(sql.distinct(joined.c.student_invs_school_year))

        inv_rates = sql.select([
                        joined.c.student_invs_student_lookup.label('student_lookup'),
                        joined.c.student_years_end_year.label('school_year'),
                        joined.c.student_years_grade,
                        joined.c.student_invs_inv_group.label('pivot_inv_group'),
                        joined.c.student_invs_description.label('description'),
                        rate_col.label('pivot_class1'),
                        rate_col.label('pivot_class2'),
                        rate_col.label('pivot_class3'),
                    ]).select_from(
                        joined
                    ).group_by(
                        joined.c.student_invs_student_lookup,
                        joined.c.student_years_end_year,
                        joined.c.student_invs_inv_group,
                        joined.c.student_invs_description,
                        joined.c.student_years_grade,
                    )

        return inv_rates
