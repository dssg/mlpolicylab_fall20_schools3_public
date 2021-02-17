import sqlalchemy as sql
from sqlalchemy import case
from  sqlalchemy import func
import sqlalchemy.sql.expression as db_expr
from sqlalchemy.types import INT, FLOAT
from schools3.data.features.features_table import FeaturesTable
from schools3.config.data import db_tables
from schools3.config.data.features import middle_school_features_config
from schools3.data.features.processors.composite_feature_processor import CompositeFeatureProcessor
from schools3.data.features.processors.pivot_processor import PivotProcessor
from schools3.data.features.processors.impute_null_processor import ImputeNullProcessor, ImputeBy

# table of middle school grade-related features derived from `clean.all_grades`
class MiddleSchoolFeatures(FeaturesTable):
    def __init__(self):
        cols = [
            sql.Column('pivot_ms_grade', INT),
            sql.Column('pivot_ms_avg_gpa', FLOAT),
        ]
        feature_processor = CompositeFeatureProcessor([
            ImputeNullProcessor(fill_unspecified=ImputeBy.MEAN)
        ])

        super(MiddleSchoolFeatures, self).__init__(
            table_name='middle_school_features',
            feature_cols=cols,
            categorical_cols=[],
            post_features_processor=feature_processor,
            pre_features_processor=PivotProcessor(
                index='student_lookup',
                columns='pivot_ms_grade',
                values='pivot_ms_avg_gpa'
            ),
            high_school_features=False
        )

    def get_data_query(self):
        all_grades = db_tables.clean_all_grades_table

        gpa_4scale = case(
            [
                (db_expr.or_(
                    all_grades.c.mark.in_(['S', 'S+', 'S-', 'A', 'A+', 'A-']),
                    all_grades.c.mark.like('9%')
                    ), 4.0),
                (db_expr.or_(
                    all_grades.c.mark.in_(['B', 'B+', 'B-']),
                    all_grades.c.mark.like('8%')
                    ), 3.0),
                (db_expr.or_(
                    all_grades.c.mark.in_(['C', 'C+', 'C-']),
                    all_grades.c.mark.like('7%')
                    ), 2.0),
                (db_expr.or_(
                    all_grades.c.mark.in_(['D', 'D+', 'D-']),
                    all_grades.c.mark.like('6%')
                    ), 1.0),
                (all_grades.c.mark.in_(['E', 'F']), 0.0)
            ],
            else_ = None
        )

        ms_grades_4scale = sql.select([
            all_grades.c.student_lookup,
            all_grades.c.grade,
            all_grades.c.course_code,
            gpa_4scale.label('gpa_4scale')
        ]).where(
            db_expr.and_(
                all_grades.c.grade >= middle_school_features_config.start_gpa_grade,
                all_grades.c.grade <= middle_school_features_config.end_gpa_grade
            )
        ).cte('ms_grades_4scale')

        ms_course_gpa = sql.select([
            ms_grades_4scale.c.student_lookup,
            ms_grades_4scale.c.grade,
            ms_grades_4scale.c.course_code,
            func.avg(ms_grades_4scale.c.gpa_4scale).label('course_avg_gpa')
        ]).where(
            ms_grades_4scale.c.gpa_4scale != None
        ).group_by(
            ms_grades_4scale.c.student_lookup,
            ms_grades_4scale.c.grade,
            ms_grades_4scale.c.course_code
        ).cte('ms_course_gpa')

        ms_gpa = sql.select([
            ms_course_gpa.c.student_lookup.label('student_lookup'),
            ms_course_gpa.c.grade.label('pivot_ms_grade'),
            func.avg(ms_course_gpa.c.course_avg_gpa).label('pivot_ms_avg_gpa')
        ]).group_by(
            ms_course_gpa.c.student_lookup,
            ms_course_gpa.c.grade
        )

        return ms_gpa
