import sqlalchemy as sql
import inflection
from sqlalchemy.types import INT, VARCHAR, FLOAT
from sqlalchemy.sql import func as db_func

from schools3.data.features.features_table import FeaturesTable
from schools3.config.data import db_tables
from schools3.config.data.features import features_config
from schools3.config.data.features import absence_features_config
from schools3.data.features.processors.impute_null_processor import ImputeNullProcessor
from schools3.data.features.processors.composite_feature_processor import CompositeFeatureProcessor


class AbsenceFeatures(FeaturesTable):
    def __init__(self):
        feature_cols = [
            sql.Column('absence_rate',FLOAT),
            sql.Column('unexcused_absence_rate',FLOAT),
            sql.Column('absence_rate_perc',FLOAT)
        ]
        super(AbsenceFeatures, self).__init__(
            table_name=inflection.underscore(AbsenceFeatures.__name__), # converts AbsenceFeatures to 'absence_features'
            feature_cols=feature_cols,
            categorical_cols=[],
            post_features_processor=CompositeFeatureProcessor([
                ImputeNullProcessor(
                    col_val_dict= absence_features_config.fill_values
                )
            ])
        )

    def get_data_query(self):
        all_snapshot= db_tables.clean_all_snapshots_table

        student_lookup= all_snapshot.c.student_lookup
        school_year= all_snapshot.c.school_year
        grade= all_snapshot.c.grade
        days_absent= all_snapshot.c.days_absent
        days_absent_unexcused= all_snapshot.c.days_absent_unexcused

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

        student_absence= sql.select([
                            student_lookup,
                            school_year,
                            grade,
                            days_absent,
                            days_absent_unexcused
                        ]).where(
                            grade >= features_config.min_grade
                        ).alias('student_absence')

        joined= sql.join(
            left= student_absence,
            right= student_years,
            onclause=sql.and_(
                        student_absence.c.student_lookup == student_years.c.student_lookup,
                        student_absence.c.school_year <= student_years.c.end_year
                    )
        )

        absence_rates = sql.select([
                        joined.c.student_absence_student_lookup.label('student_lookup'),
                        joined.c.student_years_end_year.label('school_year'),
                        joined.c.student_years_grade.label('grade'),
                        db_func.avg(joined.c.student_absence_days_absent).label('absence_rate'),
                        db_func.avg(joined.c.student_absence_days_absent_unexcused).label('unexcused_absence_rate')

                        ]).select_from(
                            joined
                        ).group_by(
                            joined.c.student_absence_student_lookup.label('student_lookup'),
                            joined.c.student_years_end_year.label('school_year'),
                            joined.c.student_years_grade.label('grade')
                        ).alias('absence_rates')

        return sql.select(
            [c for c in absence_rates.c] + 
            [db_func.percent_rank().over(
                order_by=absence_rates.c.absence_rate,
                partition_by=[absence_rates.c.school_year, absence_rates.c.grade]
            ).label('absence_rate_perc')]
        )
