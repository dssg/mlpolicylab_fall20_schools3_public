import sqlalchemy as sql
from schools3.config.data import db_config
import pandas as pd
import sqlparse
from sqlalchemy.dialects import postgresql

# a class that is able to run and print SQL queries
class Processor():
    def __init__(self, debug=False):
        self.debug = debug
        self.engine = sql.create_engine(db_config.engine_url, echo=self.debug)

    # execute the given query and return a pd.DataFrame object with result
    def query(self, query):
        return pd.read_sql(query, self.engine)

    # print out the SQL query
    def pretty_print(self, query):
        raw = query.compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": True}
            )
        print(sqlparse.format(str(raw), reindent=True))
