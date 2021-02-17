import sqlalchemy as sql
from sqlalchemy.types import INT, VARCHAR
from schools3.data.base.cohort import Cohort
from schools3.data.base.cohortable_table import CohortableTable
from schools3.data.features.processors.feature_processor import FeatureProcessor
from pandas.api.types import CategoricalDtype 

# base class for a table of features. Contains student lookup, (maybe) year and grade, and other features
class FeaturesTable(CohortableTable):
    def __init__(
            self, table_name, feature_cols, categorical_cols,
            post_features_processor: FeatureProcessor,
            pre_features_processor: FeatureProcessor = None, 
            high_school_features = True,
            debug=False
    ):

        self.base_cols = [
            sql.Column('student_lookup', INT),
        ]
        if high_school_features:
            self.base_cols += [
                sql.Column('school_year', INT),
                sql.Column('grade', INT)
            ]
        self.features_cols = feature_cols
        self.cols = self.base_cols + self.features_cols

        self.categorical_cols = categorical_cols
        self.pre_feature_processors = pre_features_processor
        self.post_feature_processors = post_features_processor
        self.high_school_features = high_school_features
        super(FeaturesTable, self).__init__(
            table_name=table_name,
            columns=self.cols,
            debug=debug
        )

        if self.pre_feature_processors:
            self._df = self.pre_feature_processors(self._df)

        duplicate_idx = self._df.duplicated(subset=[c.name for c in self.base_cols])
        if duplicate_idx.any():
            duplicate_df = self._df.duplicated()
            assert (duplicate_idx == duplicate_df).all(), 'Pre processed dataframe has different rows for same identifiers'
            self._df = self._df[~duplicate_df]

        for c in self.categorical_cols:
            if self._df[c].dropna().map(type).eq(str).all():
                self._df[c] = self._df[c].str.replace('_', ' ')
            self._df[c] = self._df[c].astype(CategoricalDtype(self._df[c].dropna().unique()))


    def process_original_df(self, df):
        return self.post_feature_processors(df)
