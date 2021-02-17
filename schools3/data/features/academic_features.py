import sqlalchemy as sql
from sqlalchemy import func, distinct
import sqlalchemy.sql.functions as db_func
import sqlalchemy.sql.expression as db_expr
from sqlalchemy.orm import aliased
from sqlalchemy.types import ARRAY, INT, VARCHAR, FLOAT
from schools3.data.features.features_table import FeaturesTable
from schools3.config.data import db_tables
from schools3.data import common_queries
from schools3.config.data.features import academic_features_config
from schools3.data.features.processors.composite_feature_processor import CompositeFeatureProcessor
from schools3.data.features.processors.impute_null_processor import ImputeNullProcessor

# table of high school grade-related features derived from `clean.all_snapshots` and `clean.high_school_gpa`
class AcademicFeatures(FeaturesTable):
    def __init__(self):
        cols = [
            sql.Column('num_classes', INT),
            sql.Column('hs_avg_gpa', FLOAT),
            sql.Column('last_year_gpa', FLOAT),
            sql.Column('pre_2_year_gpa', FLOAT),
            sql.Column('pre_3_year_gpa', FLOAT),
            sql.Column('overall_percentile', FLOAT),
            sql.Column('district_percentile', FLOAT),
            sql.Column('school_percentile', FLOAT),
        ]
        feature_processor = CompositeFeatureProcessor([
            ImputeNullProcessor(
                col_val_dict=academic_features_config.fill_values,
                fill_unspecified=False
            )
        ])
        super(AcademicFeatures, self).__init__(
            table_name='academic_features',
            feature_cols=cols,
            categorical_cols=[],
            post_features_processor=feature_processor
        )

    def get_students_grade_gpa(self):
        '''
            Returns a query that can returns a table with "grade", "district", "school_code" column to the
            high_school_gpa table
        '''
        high_school_gpa = db_tables.clean_high_school_gpa_table
        all_snapshots = db_tables.clean_all_snapshots_table

        left = sql.select([
                        all_snapshots.c.student_lookup,
                        all_snapshots.c.grade,
                        all_snapshots.c.school_year,
                        all_snapshots.c.district,
                        all_snapshots.c.school_code
                    ]).\
                    where(
                        sql.and_(
                            all_snapshots.c.grade >= 9,
                            all_snapshots.c.grade <= 12
                        )
                ).alias('a')

        right = high_school_gpa.alias('b')

        joined = sql.join(
                left=left,
                right=right,
                onclause=sql.and_(
                        left.c.student_lookup == right.c.student_lookup,
                        left.c.school_year == right.c.school_year,
                    )
                )

        overall_percentile = func.percent_rank().\
            over(order_by= joined.c.b_gpa.desc(),
                 partition_by=[joined.c.a_school_year, joined.c.a_grade])

        district_percentile = func.percent_rank().\
            over(order_by= joined.c.b_gpa.desc(),
                 partition_by=[joined.c.a_school_year, joined.c.a_grade, joined.c.a_district])

        school_percentile = func.percent_rank().\
            over(order_by= joined.c.b_gpa.desc(),
                 partition_by=[joined.c.a_school_year, joined.c.a_grade, joined.c.a_district, joined.c.a_school_code])

        return sql.select([
            joined.c.a_student_lookup,
            joined.c.a_grade,
            joined.c.a_school_year,
            joined.c.a_district,
            joined.c.a_school_code,
            joined.c.b_gpa,
            joined.c.b_num_classes,
            overall_percentile.label('overall_percentile'),
            district_percentile.label('district_percentile'),
            school_percentile.label('school_percentile')
        ]).\
        select_from(joined).\
        group_by(*list(joined.c))


    def get_data_query(self):
        hs_gpa_grades = self.get_students_grade_gpa().cte('hs_gpa_grades')


        a = aliased(hs_gpa_grades, name='a')
        b = aliased(hs_gpa_grades, name='b')
        c = aliased(hs_gpa_grades, name='c')
        d = aliased(hs_gpa_grades, name='d')

        joined = sql.join(
            left=a, right=b,
            onclause=db_expr.and_(
                a.c.student_lookup == b.c.student_lookup,
                a.c.school_year == b.c.school_year + 1
            ),
            isouter=True
        )

        joined = sql.join(
            left=joined, right=c,
            onclause=db_expr.and_(
                joined.c[a.name+'_student_lookup'] == c.c.student_lookup,
                joined.c[a.name+'_school_year'] == c.c.school_year + 2
            ),
            isouter=True
        )

        joined = sql.join(
            left=joined, right=d,
            onclause=db_expr.and_(
                joined.c[a.name+'_student_lookup'] == d.c.student_lookup,
                joined.c[a.name+'_school_year'] == d.c.school_year + 3
            ),
            isouter=True
        )

        joined_cols = [
            joined.c[a.name+'_student_lookup'].label('student_lookup'),
            joined.c[a.name+'_school_year'].label('school_year'),
            joined.c[a.name+'_grade'].label('grade'),
            joined.c[a.name+'_num_classes'],
            joined.c[a.name+'_gpa'].label('hs_avg_gpa'),
            joined.c[b.name+'_gpa'].label('last_year_gpa'),
            joined.c[c.name+'_gpa'].label('pre_2_year_gpa'),
            joined.c[d.name+'_gpa'].label('pre_3_year_gpa'),
            joined.c[a.name+'_overall_percentile'],
            joined.c[a.name+'_district_percentile'],
            joined.c[a.name+'_school_percentile']
        ]

        final = sql.select(
            joined_cols
        ).\
        select_from(
            joined
        ).\
        distinct(
            joined.c[a.name+'_student_lookup'].label('student_lookup'),
            joined.c[a.name+'_school_year'].label('school_year'),
            joined.c[a.name+'_grade'].label('grade')
        ).\
        order_by(
            joined.c[a.name+'_student_lookup'],
            sql.desc(joined.c[a.name+'_school_year'])
        )

        return final
