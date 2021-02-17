import sqlalchemy as sql
import inflection
from sqlalchemy.types import INT, VARCHAR, FLOAT
from sqlalchemy.sql import func as db_func

from schools3.data.features.features_table import FeaturesTable
from schools3.config.data import db_tables
from schools3.config.data.features import features_config
from schools3.config.data.features import discipline_incident_rate_features_config as config
from schools3.data.features.processors.impute_null_processor import ImputeNullProcessor
from schools3.data.features.processors.composite_feature_processor import CompositeFeatureProcessor


class DisciplineIncidentRateFeatures(FeaturesTable):
    def __init__(self):
        feature_cols = [
            sql.Column('discipline_incident_rate',FLOAT),
            sql.Column('discipline_incident_rate_perc',FLOAT)
        ]
        super(DisciplineIncidentRateFeatures, self).__init__(
            table_name=inflection.underscore(DisciplineIncidentRateFeatures.__name__), 
            feature_cols=feature_cols,
            categorical_cols=[],
            post_features_processor=CompositeFeatureProcessor([
                ImputeNullProcessor(
                    col_val_dict= config.fill_values
                )
            ])
        )

    def get_data_query(self):
        all_snapshot= db_tables.clean_all_snapshots_table

        student_lookup= all_snapshot.c.student_lookup
        school_year= all_snapshot.c.school_year
        grade= all_snapshot.c.grade
        discipline_incidents= all_snapshot.c.discipline_incidents

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

        student_discipline= sql.select([
                            student_lookup,
                            school_year,
                            grade,
                            discipline_incidents
                        ]).where(
                            grade >= features_config.min_grade
                        ).alias('student_discipline')

        joined= sql.join(
            left= student_discipline,
            right= student_years,
            onclause=sql.and_(
                        student_discipline.c.student_lookup == student_years.c.student_lookup,
                        student_discipline.c.school_year <= student_years.c.end_year
                    )
        )

        discipline_incident_rates= sql.select([
                        joined.c.student_discipline_student_lookup.label('student_lookup'),
                        joined.c.student_years_end_year.label('school_year'),
                        joined.c.student_years_grade.label('grade'),
                        db_func.avg(joined.c.student_discipline_discipline_incidents).label('discipline_incident_rate'),
                        db_func.percent_rank().over(
                            order_by=db_func.avg(joined.c.student_discipline_discipline_incidents),
                            partition_by=[joined.c.student_years_end_year, joined.c.student_years_grade]
                        ).label('discipline_incident_rate_perc')
                    ]).select_from(
                        joined
                    ).group_by(
                        joined.c.student_discipline_student_lookup.label('student_lookup'),
                        joined.c.student_years_end_year.label('school_year'),
                        joined.c.student_years_grade.label('grade')
                    )

        return discipline_incident_rates
