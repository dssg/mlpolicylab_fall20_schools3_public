from abc import ABC, abstractmethod
from typing import List
import sqlalchemy as sql
import pandas as pd
from tqdm import tqdm
import numpy as np
import requests
from schools3.data.base.schools_table import SchoolsTable
from schools3.config.data import db_config
import schools3.config.data.features.acs_features_config as config

class ACSTable(SchoolsTable):
    def __init__(self, feature_group_name, column_names):
        self.column_names = column_names

        cols = [
            sql.Column('zip_code', sql.VARCHAR),
        ]
        cols += [sql.Column(col, sql.FLOAT) for col in column_names]

        super(ACSTable, self).__init__(
            table_name=feature_group_name,
            columns=cols,
            schema_name=config.schema_name,
        )

    def _get_df(self):
        '''
            Download all features of the feature_group specified and return a
            dataframe that stores these features WITHOUT ANY FURTHER PROCESSING.
            The first column of the dataframe should be zip_code, so it can be
            used in joining
        '''
        zipcode_df, all_zipcodes = self.extract_zipcode()
        acs_df = self.get_acs_data(self.column_names)

        # merge dataframes
        return self.merge_acs_zipcode(acs_df, zipcode_df, all_zipcodes)

    def extract_zipcode(self):
        zipcode_df = pd.read_csv(config.zipcodes_fname, dtype=config.zipcodes_dtypes)

        # drop rows that aren't from Ohio, also drop columns that we don't need
        check_state = lambda x: x.startswith(str(config.state_fips))
        zipcode_df = zipcode_df[zipcode_df['TRACT'].map(check_state)]
        zipcode_df = zipcode_df.drop(columns=['RES_RATIO', 'BUS_RATIO', 'OTH_RATIO'])

        all_zipcodes = zipcode_df['ZIP'].unique()
        zipcode_df = zipcode_df.set_index(['ZIP'])

        return zipcode_df, all_zipcodes

    def get_acs_data(self, columns):
        '''
            given column names, query APIs to return raw data in pandas
            dataframe style
        '''
        base_url = config.raw_data_base_link
        data = None
        counties_fips, _ = config.get_counties()
        for c in tqdm(counties_fips):
            parameters = {
                'for': config.for_param,
                'in': config.get_in_param(config.state_fips, c),
                'get': ','.join(columns),
                'key': config.key
            }
            if not data:
                data = requests.get(base_url, params=parameters).json()
            else:
                data += requests.get(base_url, params=parameters).json()[1:]

        df = pd.DataFrame(data[1:], columns=data[0])
        for c in columns:
            df[c] = df[c].astype(np.float)

        return df

    def merge_acs_zipcode(self, acs_df, zipcode_df, all_zipcodes):
        # convert acs dataframe to use the same tract formatting as the zipcode dataframe
        acs_df['tract_id'] = acs_df.apply(lambda row: row['state'] + row['county'] + row['tract'], axis=1)
        acs_df = acs_df.drop(columns=['state', 'county', 'tract', 'block group'])
        acs_df = acs_df.groupby(['tract_id']).sum()

        # reduce acs dataframe to zipcode-level information using weighted sums
        final_df = pd.DataFrame()
        for zipcode in all_zipcodes:
            zipcode_rows = zipcode_df.loc[zipcode]
            zipcode_rows = [x for _, x in zipcode_rows.iterrows()] if hasattr(zipcode_rows, 'iterrows') else [zipcode_rows]

            tracts = [row['TRACT'] for row in zipcode_rows]
            ratios = [row['TOT_RATIO'] for row in zipcode_rows]

            weighted_feats = acs_df.loc[tracts].mul(ratios, axis=0)
            weighted_feats['zipcode'] = zipcode
            weighted_feats = weighted_feats.groupby(['zipcode']).sum()

            final_df = final_df.append(weighted_feats)

        return final_df.reset_index()