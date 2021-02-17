from abc import ABC, abstractmethod
import sqlalchemy as sql
from typing import List
import pandas as pd
from schools3.data.base.schools_table import SchoolsTable
from schools3.data.base.cohort import Cohort

# base class for feature and label tables.
# Defines interface for joining a table of information with a cohort
class CohortableTable(SchoolsTable):
    def __init__(self, table_name, columns: List[sql.Column], debug=False):
        super(CohortableTable, self).__init__(table_name, columns, debug=debug)

        self._df = self._get_df()
        assert sorted(list(self._df.columns)) == sorted([c.name for c in self.columns])

    # merges the Dataframe for this object after joining with a given cohort
    def get_for(self, cohort: Cohort, keep_cohort_cols=False):
        c = cohort.get_cohort()
        common_cols = c.columns.intersection(self._df.columns)
        cohort_df = self._df.merge(c, how='inner', on=list(common_cols))
        cohort_df = self.process_original_df(cohort_df)
        if keep_cohort_cols:
            return cohort_df
        else:
            return cohort_df.drop(c.columns.difference(common_cols), axis=1)

    # returns the Dataframe for this object
    def get_all(self):
        return self.process_original_df(self._df.copy())

    # runs some feature processors on this object's Dataframe and returns it
    @abstractmethod
    def process_original_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _get_df(self):
        return self.query(self.get_data_query())
