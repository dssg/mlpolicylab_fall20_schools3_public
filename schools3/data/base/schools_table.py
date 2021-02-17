from abc import ABC, abstractmethod
from typing import List
import io
import sqlalchemy as sql

from schools3.data.base.processor import Processor
from schools3.config.data import db_config

# a wrapper for a table of data. Specified as either a Dataframe or a SQL query
class SchoolsTable(Processor):
    def __init__(self, table_name, columns: List[sql.Column], schema_name = db_config.write_schema_name, debug=False):
        super(SchoolsTable, self).__init__(debug=debug)

        self.table_name = table_name
        self.schema = schema_name
        self.columns = columns
        self.meta = sql.MetaData()
        self.table = sql.Table(
            table_name, self.meta,
            *columns,
            schema=self.schema
        )

    # write this table to the database if it does not exist
    def maybe_create(self):
        if not self.engine.has_table(self.table_name, schema=self.schema):
            self.table.create(self.engine)

    # abstract method to get SQL query for this table
    @abstractmethod
    def get_data_query(self):
        pass

    # abstract method to get the Dataframe for this table
    @abstractmethod
    def _get_df(self):
        pass

    # writes data into the database using a Dataframe
    def insert_from_df(self, use_native_copy=True):
        df = self._get_df()
        table_name = self.table_name
        schema = self.schema

        assert not self.engine.has_table(table_name, schema=schema), \
            f'Table {schema}.{table_name} already exists. Quitting'

        if use_native_copy:
            df[:0].to_sql(
                table_name, self.engine, schema=schema,
                index=False
            ) # create raw skeleton of table

            data_buf = io.StringIO()
            df.to_csv(data_buf, index=False, header=False)
            data_buf.seek(0)

            conn = self.engine.raw_connection()
            ps_curs = conn.cursor()

            ps_curs.copy_from(
                data_buf,
                table=f'{schema}.{table_name}',
                sep=',',
                null=''
            )

            conn.commit()
            ps_curs.close()

        else:
            df.to_sql(
                table_name, self.engine, schema=schema,
                index=False
            )

    # writes data into the database using a query
    def insert_from_query(self):
        data_query = self.get_data_query()

        assert len(data_query.c) == len(self.columns),\
            'number of columns in data query do not match' +\
            'the number specifed in __init__'

        self.maybe_create()
        ins = self.table.insert().from_select(data_query.c, data_query)
        with self.engine.begin() as conn:
            conn.engine.execute(ins)
