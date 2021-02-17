from schools3.data.base.processor import Processor
import sqlalchemy as sql
from schools3.config.data import db_tables

# a table of all the student_lookups in a cohort, which is specified by grade and years
class Cohort(Processor):
    def __init__(self, grade: int, years: list, debug=False):
        super(Cohort, self).__init__(debug)
        self.grade = grade
        self.years = years

        self.__cohort = self.query(self.get_cohort_query())
        self.__cohort.drop_duplicates(inplace=True, ignore_index=True)
        # self.__cohort.set_index(
        #     list(self.__cohort.columns),
        #     inplace=True,
        #     verify_integrity=True
        # )

    # get string representation of the cohort
    def get_identifier(self):
        return str(self.grade).zfill(2) + '_' + '_'.join([str(y) for y in self.years])

    # returns the cohort as a Dataframe
    def get_cohort(self):
        return self.__cohort

    # returns the query to get the cohort
    def get_cohort_query(self):
        all_snapshots = db_tables.clean_all_snapshots_table
        return sql.select([
            sql.cast(all_snapshots.c.student_lookup, sql.INTEGER).label('student_lookup'),
            all_snapshots.c.school_year,
            all_snapshots.c.grade,
        ]).where(
            sql.and_(
                all_snapshots.c.grade == self.grade,
                all_snapshots.c.school_year.in_(self.years)
            )
        )
