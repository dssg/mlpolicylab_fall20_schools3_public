import sqlalchemy as sql
from sqlalchemy.types import VARCHAR
from sqlalchemy.sql import func as db_func
from schools3.config.data.features import features_config
from schools3.data.features.features_table import FeaturesTable
from schools3.data.features.processors.pivot_processor import PivotProcessor


class PivotBlockFeatures(FeaturesTable):
    def __init__(
        self, table_name, categorical_cols, post_features_processor, data_table,
        blocking_col, index_cols_dict=None,
        index=['student_lookup', 'school_year', 'grade'],
    ):
        self.value_col_name = 'pivotable_rate'

        feature_cols = [
            sql.Column(blocking_col.name, VARCHAR),
            sql.Column(self.value_col_name, VARCHAR),
        ]

        self.blocking_col = blocking_col
        self.data_table = data_table
        if index_cols_dict is None:
            index_cols_dict = {
                'student_lookup': self.data_table.c.student_lookup,
                'school_year': self.data_table.c.school_year,
                'grade': self.data_table.c.grade
            }
        self.index_cols_dict = index_cols_dict

        super(PivotBlockFeatures, self).__init__(
            table_name=table_name,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            post_features_processor=post_features_processor,
            pre_features_processor=PivotProcessor(
                    index=index,
                    columns=blocking_col.name,
                    values=self.value_col_name
                )
        )

    def get_data_query(self):
        # FIXME: Make end year go upto the last year on record
        index_cols_dict = self.index_cols_dict

        student_years = sql.select(
                            list(index_cols_dict.values())
                        ).distinct(
                            *list(index_cols_dict.values())
                        ).where(
                            index_cols_dict['grade'] >= 9
                        ).alias('student_years')

        student_block = sql.select(
                            list(index_cols_dict.values()) +
                            [self.blocking_col]
                        ).where(
                            index_cols_dict['grade'] >= features_config.min_grade
                        ).alias('student_block')

        joined = sql.join(
            left=student_block,
            right=student_years,
            onclause=sql.and_(
                        student_block.c.student_lookup == student_years.c.student_lookup,
                        student_block.c.school_year <= student_years.c.school_year
                    )
        )

        value_col = db_func.count() * 1.0 / db_func.count(sql.distinct(joined.c.student_block_school_year))

        inv_rates = sql.select([
                        joined.c.student_block_student_lookup.label('student_lookup'),
                        joined.c.student_years_school_year.label('school_year'),
                        joined.c.student_years_grade,
                        joined.c['student_block_' + self.blocking_col.name].label(self.blocking_col.name),
                        value_col.label(self.value_col_name),
                    ]).select_from(
                        joined
                    ).group_by(
                        joined.c.student_block_student_lookup,
                        joined.c.student_years_school_year,
                        joined.c.student_years_grade,
                        joined.c['student_block_' + self.blocking_col.name],
                    )

        return inv_rates
