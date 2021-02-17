import pandas as pd
import sqlalchemy as sql
import sqlalchemy.sql.functions as db_func
import sqlalchemy.sql.expression as db_expr
from sqlalchemy.orm import aliased
from sqlalchemy.types import ARRAY, INT, VARCHAR, FLOAT
from schools3.config.data import db_tables
from schools3.config.data.features import features_config
from schools3.config.data.features import snapshot_features_config
from schools3.data.features.features_table import FeaturesTable
from schools3.data.features.processors.composite_feature_processor import CompositeFeatureProcessor
from schools3.data.features.processors.impute_null_processor import ImputeNullProcessor
from schools3.data.features.processors.categorical_feature_processor import CategoricalFeatureProcessor
from schools3.data.features.processors.replace_nullish_processor import ReplaceNullishProcessor

# table of features derived from `clean.all_snapshots` in the database

# Not all features we use that are based on `clean.all_snapshots` are computed in this file
# However, this table contains many features that can be directly read off or are simple to compute
class SnapshotFeatures(FeaturesTable):
    def __init__(self):
        cols = [
            sql.Column('gender', VARCHAR),
            sql.Column('ethnicity', VARCHAR),
            sql.Column('school_name', VARCHAR),
            sql.Column('district', VARCHAR),
            sql.Column('disability', VARCHAR),
            sql.Column('disadvantagement', VARCHAR),
            sql.Column('economic_disadvantagement', INT),
            sql.Column('academic_disadvantagement', INT),
            sql.Column('limited_english', VARCHAR),
            sql.Column('discipline_incidents', INT),
            sql.Column('pre_1_year_discipline_incidents', INT),
            sql.Column('pre_2_year_discipline_incidents', INT),
            sql.Column('days_absent', FLOAT),
            sql.Column('pre_1_year_days_absent', FLOAT),
            sql.Column('pre_2_year_days_absent', FLOAT),
            sql.Column('age', INT),
            sql.Column('num_transfers', INT),
            sql.Column('cumul_discipline_incidents', INT),
        ]

        feature_processor = CompositeFeatureProcessor([
            ReplaceNullishProcessor(
                column_list=snapshot_features_config.replace_nullish_columns
            ),
            ImputeNullProcessor(
                col_val_dict=snapshot_features_config.fill_values,
                col_flag_set=snapshot_features_config.impute_flag_columns
            ),
            CategoricalFeatureProcessor(
                column_list=snapshot_features_config.categorical_columns
            )
        ])
        super(SnapshotFeatures, self).__init__(
            table_name='snapshot_features',
            feature_cols=cols,
            categorical_cols=snapshot_features_config.categorical_columns,
            post_features_processor=feature_processor
        )

    def get_data_query(self):
        all_snapshots           = db_tables.clean_all_snapshots_table

        student_lookup          = all_snapshots.c.student_lookup
        school_year             = all_snapshots.c.school_year
        grade                   = all_snapshots.c.grade
        gender                  = all_snapshots.c.gender
        ethnicity               = all_snapshots.c.ethnicity
        school_name             = all_snapshots.c.school_name
        district                = all_snapshots.c.district
        disability              = all_snapshots.c.disability
        disadvantagement        = all_snapshots.c.disadvantagement
        limited_english         = all_snapshots.c.limited_english
        discipline_incidents    = all_snapshots.c.discipline_incidents
        days_absent             = all_snapshots.c.days_absent
        birth_date              = all_snapshots.c.birth_date

        age = school_year - sql.cast(sql.func.substr(sql.cast(birth_date, VARCHAR), 1, 4), INT)
        economic_disadvantagement = sql.case(
            [(all_snapshots.c.disadvantagement.in_(['economic', 'both']), 1),
             (all_snapshots.c.disadvantagement.in_(['academic', 'none']), 0)],
            else_ = None)
        academic_disadvantagement = sql.case(
            [(all_snapshots.c.disadvantagement.in_(['academic', 'both']), 1),
             (all_snapshots.c.disadvantagement.in_(['economic', 'none']), 0)],
            else_ = None)
        

        # get a single row for each (student_lookup, school_year, grade) pair
        snapshots = sql.select([
                student_lookup,
                school_year,
                grade,
                gender,
                ethnicity,
                school_name,
                district,
                disability,
                disadvantagement,
                economic_disadvantagement.label('economic_disadvantagement'),
                academic_disadvantagement.label('academic_disadvantagement'),
                limited_english,
                discipline_incidents,
                days_absent,
                age.label('age'),
            ]).\
            distinct(student_lookup, school_year, grade).\
            where(
                student_lookup != None,
            ).\
            order_by(
                student_lookup,
                sql.desc(school_year)
            ).cte('snapshots_temp_a')

        # join on previous years to get features from the past for same student_lookup
        snapshots = self.join_history_feats(snapshots)

        return snapshots
    
    def join_history_feats(self, snapshots):
        a = aliased(snapshots, name='snapshot_history_a')
        b = aliased(snapshots, name='snapshot_history_b')
        c = aliased(snapshots, name='snapshot_history_c')
        d = aliased(snapshots, name='snapshot_history_d')
        
        joined = sql.join(
            left=a, right=b,
            onclause=db_expr.and_(
                a.c.student_lookup == b.c.student_lookup,
                a.c.school_year >= b.c.school_year,
                b.c.grade >= features_config.min_grade
            ),
            isouter=True
        )
        
        joined = sql.join(
            left=joined, right=c,
            onclause=db_expr.and_(
                joined.c[a.name+'_student_lookup'] == c.c.student_lookup,
                joined.c[a.name+'_school_year'] == c.c.school_year + 1,
                joined.c[a.name+'_grade'] == c.c.grade + 1
            ),
            isouter=True
        )
        
        joined = sql.join(
            left=joined, right=d,
            onclause=db_expr.and_(
                joined.c[a.name+'_student_lookup'] == d.c.student_lookup,
                joined.c[a.name+'_school_year'] == d.c.school_year + 2,
                joined.c[a.name+'_grade'] == d.c.grade + 2
            ),
            isouter=True
        )

        num_transfers = db_func.count(sql.distinct(joined.c[b.name+'_school_name'])) - 1
        num_transfers = sql.case([(num_transfers < 0, 0)], else_=num_transfers)   # special case for nulls

        cumul_discipline_incidents = db_func.sum(joined.c[b.name+'_discipline_incidents'])

        joined_a_cols = [
            joined.c[a.name+'_student_lookup'],
            joined.c[a.name+'_school_year'],
            joined.c[a.name+'_grade'],
            joined.c[a.name+'_gender'],
            joined.c[a.name+'_ethnicity'],
            joined.c[a.name+'_school_name'],
            joined.c[a.name+'_district'],
            joined.c[a.name+'_disability'],
            joined.c[a.name+'_disadvantagement'],
            joined.c[a.name+'_economic_disadvantagement'],
            joined.c[a.name+'_academic_disadvantagement'],
            joined.c[a.name+'_limited_english'],
            joined.c[a.name+'_discipline_incidents'],
            joined.c[c.name+'_discipline_incidents'].label('pre_1_year_discipline_incidents'),
            joined.c[d.name+'_discipline_incidents'].label('pre_2_year_discipline_incidents'),
            joined.c[a.name+'_days_absent'],
            joined.c[c.name+'_days_absent'].label('pre_1_year_days_absent'),
            joined.c[d.name+'_days_absent'].label('pre_2_year_days_absent'),
            joined.c[a.name+'_age'],
        ]

        return sql.select(
            joined_a_cols + 
            [
                num_transfers.label('num_transfers'),
                cumul_discipline_incidents.label('cumul_discipline_incidents')
            ]
        ).\
        select_from(joined).\
        where(
            joined.c[a.name+'_grade'] >= 9
        ).\
        group_by(
            *joined_a_cols
        ).\
        order_by(
            joined.c[a.name+'_student_lookup'],
            sql.desc(joined.c[a.name+'_school_year'])
        )