import sqlalchemy as sql
from sqlalchemy.types import INT, VARCHAR

from schools3.data.base.cohortable_table import CohortableTable

# base class for a table of labels. Contains two columns: student lookup and label
class LabelsTable(CohortableTable):
    def __init__(self, table_name, debug=False):
        cols = [
            sql.Column('student_lookup', INT),
            sql.Column('label', VARCHAR)
        ]

        super(LabelsTable, self).__init__(
            table_name=table_name,
            columns=cols,
            debug=debug
        )
