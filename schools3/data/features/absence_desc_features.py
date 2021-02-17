import sqlalchemy as sql
import inflection
from schools3.config.data import db_tables
from schools3.data.features.pivot_block_features import PivotBlockFeatures
from schools3.data.features.processors.composite_feature_processor import CompositeFeatureProcessor
from schools3.data.features.processors.impute_null_processor import ImputeNullProcessor
from schools3.data import common_queries


class AbsenceDescFeatures(PivotBlockFeatures):
    def __init__(self):
        joined_absences = self.get_joined_absences()

        super(AbsenceDescFeatures, self).__init__(
            table_name=inflection.underscore(AbsenceDescFeatures.__name__),
            categorical_cols=[],
            post_features_processor=CompositeFeatureProcessor([
                ImputeNullProcessor(fill_unspecified=0, col_flag_set=False)
            ]),
            data_table=joined_absences,
            blocking_col=joined_absences.c.absence_desc,
        )

    def get_joined_absences(self):
        all_absences = db_tables.clean_all_absences_table
        absence_desc_table = sql.select([
            all_absences.c.student_lookup,
            all_absences.c.grade,
            all_absences.c.absence_desc,
        ]).alias('absence_desc_table')
        students = common_queries.get_snapshot_students(hs_only=False).alias('students')

        joined = sql.join(
            left=students,
            right=absence_desc_table,
            onclause=sql.and_(
                        students.c.student_lookup == absence_desc_table.c.student_lookup,
                        students.c.grade == absence_desc_table.c.grade,
                    )
        )

        return sql.select([
            joined.c.students_student_lookup.label('student_lookup'),
            joined.c.students_school_year.label('school_year'),
            joined.c.students_grade.label('grade'),
            joined.c.absence_desc_table_absence_desc.label('absence_desc'),
        ]).select_from(
            joined
        ).alias('joined_absences')
