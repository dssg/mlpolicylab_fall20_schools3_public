import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy as sql
from sqlalchemy.sql import func as sql_func
import seaborn as sns

from schools3.data.explorers.explorer import Explorer
from schools3.data import common_queries
from schools3.config.data import db_tables
from schools3.config import global_config


class TemporalExplorer(Explorer):
    def __init__(self, debug=False):
        super(TemporalExplorer, self).__init__(debug)

    def get_temporal_gpa_query(self, group_by_grade=True):
        hs_gpa_grades = common_queries.get_students_grade_gpa().cte('hs_gpa_grades')
        select_cols = [
            hs_gpa_grades.c.school_year,
            sql_func.avg(hs_gpa_grades.c.gpa).label('avg_gpa')
        ]
        group_by_cols = [hs_gpa_grades.c.school_year]

        if group_by_grade:
            group_by_cols.append(hs_gpa_grades.c.grade)
            select_cols.append(hs_gpa_grades.c.grade)

        return sql.select(select_cols).group_by(*group_by_cols)

    def get_year_grade_plots(self, df, value_col, plot_type='multiline'):
        assert plot_type in ['multiline', 'heatmap']
        pivoted = df.pivot('school_year', 'grade', value_col)
        if plot_type == 'multiline':
            return sns.lineplot(data=pivoted)
        else:
            return sns.heatmap(data=pivoted, annot=True)

    def get_year_line_plots(self, df, value_col):
        assert df.school_year.is_unique, 'school_year must not have duplicates'
        indexed = df.set_index('school_year')
        return sns.lineplot(x=indexed.index, y=indexed[value_col])

    def get_temporal_absenteeism_di_query(self, group_by_grade=True):
        snapshots = db_tables.clean_all_snapshots_table

        select_cols = [
            snapshots.c.school_year,
            sql_func.avg(snapshots.c.days_absent).label('avg_days_absent'),
            sql_func.avg(snapshots.c.discipline_incidents).label('avg_discipline_incidents')
        ]
        group_by_cols = [snapshots.c.school_year]

        if group_by_grade:
            corrected_grade = sql.case(
                            [snapshots.c.grade > 12, 12],
                            else_=snapshots.c.grade
                        )
            select_cols.append(corrected_grade.label('grade'))
            group_by_cols.append(corrected_grade)

        return sql.select(
            select_cols
        ).where(
            sql.and_(
                snapshots.c.grade >= 9,
                snapshots.c.grade <= 12
            )
        ).group_by(
            *group_by_cols
        )
