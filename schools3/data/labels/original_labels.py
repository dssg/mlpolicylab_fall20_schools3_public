import sqlalchemy as sql
import sqlalchemy.sql.functions as db_func
import sqlalchemy.sql.expression as db_expr
from sqlalchemy.types import INT, VARCHAR

from schools3.data.labels.labels_table import LabelsTable
from schools3.config.data import db_tables
from schools3.config.data.labels import original_labels_config

# table of label information, computed based on the definition of graduation we proposed
class OriginalLabels(LabelsTable):
    def __init__(self):
        super(OriginalLabels, self).__init__(
          table_name='original_labels',
        )

    def process_original_df(self, df):
        return self.get_binary_labels(df)

    def get_binary_labels(self, df):
        df['label'] = df['label'].apply(lambda x: original_labels_config.labels_dict[x])
        return df

    def get_data_query(self):
        all_snapshots              = db_tables.clean_all_snapshots_table
        high_school_gpa            = db_tables.clean_high_school_gpa_table
        grade_info                 = self.get_grade_info(all_snapshots).cte('grade_info')
        on_time_grads              = self.get_on_time_grads(grade_info).cte('on_time_grads')
        late_grads                 = self.get_late_grads(grade_info).cte('late_grads')
        dropouts                   = self.get_dropouts(grade_info).cte('dropouts')
        end_year                   = self.get_end_year(high_school_gpa).cte('end_year')
        end_grade                  = self.get_end_grade(all_snapshots, end_year).cte('end_grade')
        final_gpa                  = self.get_final_gpa(high_school_gpa, end_grade).cte('final_gpa')
        low_gpa_thresh             = self.get_low_gpa_thresh(final_gpa).cte('low_gpa_thresh')
        high_gpa_thresh            = self.get_high_gpa_thresh(final_gpa).cte('high_gpa_thresh')
        unclassified_gpa_seniors   = self.get_unclassified_gpa_seniors(final_gpa, on_time_grads, late_grads, dropouts).cte('unclassified_gpa_seniors')
        unconfirmed_low_gpa        = self.get_unconfirmed_low_gpa(unclassified_gpa_seniors, low_gpa_thresh).cte('unconfirmed_low_gpa')
        unconfirmed_high_gpa       = self.get_unconfirmed_high_gpa(unclassified_gpa_seniors, high_gpa_thresh).cte('unconfirmed_high_gpa')
        unclassified               = self.get_unclassified(all_snapshots, on_time_grads, late_grads, dropouts, unconfirmed_high_gpa, unconfirmed_low_gpa).cte('unclassified')

        u = sql.union(
            on_time_grads.select(),
            late_grads.select(),
            dropouts.select(),
            unconfirmed_high_gpa.select(),
            unconfirmed_low_gpa.select(),
            unclassified.select()
        ).alias('all_labels')

        return sql.select([
            sql.cast(u.c.student_lookup, INT).label('student_lookup'),
            u.c.label
        ])

    def get_grade_info(self, all_snapshots):
        grade           = all_snapshots.c.grade
        graduation_date = all_snapshots.c.graduation_date
        school_year     = all_snapshots.c.school_year
        status          = all_snapshots.c.status
        student_lookup  = all_snapshots.c.student_lookup
        withdraw_reason = all_snapshots.c.withdraw_reason

        end_grade = sql.case([(db_func.max(grade) > 12, 12)], else_=db_func.max(grade))
        start_grade = sql.case([(db_func.min(grade) > 12, 12)], else_=db_func.min(grade))

        return \
            sql.select([
                student_lookup,
                (end_grade - start_grade).label('num_grades'),
                (db_func.max(school_year) - db_func.min(school_year)).label('num_hs_years'),
                start_grade.label('start_grade'),
                end_grade.label('end_grade'),
                db_func.min(school_year).label('start_year'),
                db_func.max(school_year).label('end_year'),
                db_func.min(graduation_date).label('graduation_date'),
                db_func.array_agg(sql.distinct(status)).label('statuses'),
                db_func.array_agg(sql.distinct(withdraw_reason)).label('withdraw_reasons'),
                db_func.array_agg(sql.distinct(sql.func.substr(withdraw_reason, 1, 7))).label('withdraw_reasons_short'),
            ]).\
            where(
                grade >= 9
            ).\
            group_by(
                student_lookup
            )

    def get_on_time_grads(self, grade_info):
        end_grade               = grade_info.c.end_grade
        end_year                = grade_info.c.end_year
        graduation_date         = grade_info.c.graduation_date
        num_grades              = grade_info.c.num_grades
        num_hs_years            = grade_info.c.num_hs_years
        statuses                = grade_info.c.statuses
        student_lookup          = grade_info.c.student_lookup
        withdraw_reasons        = grade_info.c.withdraw_reasons
        withdraw_reasons_short  = grade_info.c.withdraw_reasons_short

        return \
            sql.select([
                sql.distinct(student_lookup).label('student_lookup'),
                sql.literal(original_labels_config.on_time_grad).label('label')
            ]).\
            where(
                db_expr.and_(
                    db_expr.or_(
                        db_expr.and_(
                            graduation_date != None,
                            sql.cast(sql.func.substr(sql.cast(graduation_date, VARCHAR), 1, 4), INT) == end_year
                        ),
                        sql.func.array_position(withdraw_reasons, 'graduate') != None,
                        sql.func.array_position(statuses, 'graduate') != None,
                    ),
                    sql.func.array_position(withdraw_reasons_short, 'dropout') == None,
                    end_grade == 12,
                    num_grades == num_hs_years
                )
            )

    def get_late_grads(self, grade_info):
        end_year                = grade_info.c.end_year
        graduation_date         = grade_info.c.graduation_date
        num_grades              = grade_info.c.num_grades
        num_hs_years            = grade_info.c.num_hs_years
        student_lookup          = grade_info.c.student_lookup
        withdraw_reasons_short  = grade_info.c.withdraw_reasons_short

        return \
            sql.select([
                sql.distinct(student_lookup).label('student_lookup'),
                sql.literal(original_labels_config.late_grad).label('label')
            ]).\
            where(
                db_expr.and_(
                    db_expr.or_(
                        graduation_date == None,
                        sql.cast(sql.func.substr(sql.cast(graduation_date, VARCHAR), 1, 4), INT) == end_year
                    ),
                    sql.func.array_position(withdraw_reasons_short, 'dropout') == None,
                    num_grades < num_hs_years,
                    num_grades > 0
                )
            )

    def get_dropouts(self, grade_info):
        student_lookup          = grade_info.c.student_lookup
        withdraw_reasons_short  = grade_info.c.withdraw_reasons_short

        return \
            sql.select([
                sql.distinct(student_lookup).label('student_lookup'),
                sql.literal(original_labels_config.dropout).label('label')
            ]).\
            where(
                sql.func.array_position(withdraw_reasons_short, 'dropout') != None,
            )

    def get_end_year(self, high_school_gpa):
        student_lookup  = high_school_gpa.c.student_lookup
        school_year     = high_school_gpa.c.school_year
        num_classes     = high_school_gpa.c.num_classes

        return \
            sql.select([
                student_lookup,
                db_func.max(school_year).label('end_year'),
                db_func.sum(num_classes).label('total_classes'),
            ]).\
            group_by(
                student_lookup
            )

    def get_end_grade(self, all_snapshots, end_year):
        a = sql.select([
                all_snapshots.c.student_lookup.label('student_lookup'),
                all_snapshots.c.grade.label('grade'),
                all_snapshots.c.school_year.label('school_year'),
            ]).\
            where(
                all_snapshots.c.grade >= 9,
            ).\
            group_by(
                all_snapshots.c.student_lookup,
                all_snapshots.c.grade,
                all_snapshots.c.school_year,
            ).cte('a')

        joined = sql.join(
                    left=a, right=end_year,
                    onclause=db_expr.and_(
                        a.c.student_lookup == end_year.c.student_lookup,
                        a.c.school_year == end_year.c.end_year,
                    ),
                    isouter=True
                )

        return \
            sql.select([
                joined.c[a.name+'_student_lookup'],
                joined.c[a.name+'_grade'],
                joined.c[end_year.name+'_end_year'],
                joined.c[end_year.name+'_total_classes'],
            ]).\
            select_from(joined).\
            group_by(
                joined.c[a.name+'_student_lookup'],
                joined.c[a.name+'_grade'],
                joined.c[end_year.name+'_end_year'],
                joined.c[end_year.name+'_total_classes'],
            )

    def get_final_gpa(self, high_school_gpa, end_grade):
        joined = sql.join(
                    left=high_school_gpa, right=end_grade,
                    onclause=db_expr.and_(
                        high_school_gpa.c.student_lookup == end_grade.c.student_lookup,
                        high_school_gpa.c.school_year == end_grade.c.end_year,
                    ),
                )

        return \
            sql.select([
                joined.c['clean_'+high_school_gpa.name+'_student_lookup'],
                joined.c['clean_'+high_school_gpa.name+'_school_year'],
                joined.c['clean_'+high_school_gpa.name+'_gpa'],
                joined.c[end_grade.name+'_grade'],
                joined.c[end_grade.name+'_total_classes'],
            ]).\
            select_from(joined)

    def get_low_gpa_thresh(self, final_gpa):
        return \
            sql.select([
                sql.func.percentile_cont(original_labels_config.low_gpa_percentile).within_group(final_gpa.c.gpa)
            ]).\
            where(
                final_gpa.c.grade >= 12
            )

    def get_high_gpa_thresh(self, final_gpa):
        return \
            sql.select([
                sql.func.percentile_cont(original_labels_config.high_gpa_percentile).within_group(final_gpa.c.gpa)
            ]).\
            where(
                final_gpa.c.grade >= 12
            )

    def get_unclassified_gpa_seniors(self, final_gpa, on_time_grads, late_grads, dropouts):
        on_time_lookups = sql.select([on_time_grads.c.student_lookup])
        late_lookups    = sql.select([late_grads.c.student_lookup])
        dropout_lookups = sql.select([dropouts.c.student_lookup])

        u = sql.union(on_time_lookups, late_lookups, dropout_lookups)

        return \
            sql.select([
                final_gpa.c.student_lookup,
                final_gpa.c.gpa
            ]).\
            where(
                db_expr.and_(
                    final_gpa.c.student_lookup.notin_(u),
                    final_gpa.c.grade >= 12
                )
            )

    def get_unconfirmed_low_gpa(self, unclassified_gpa_seniors, low_gpa_thresh):
        unclassified_lookup = unclassified_gpa_seniors.c.student_lookup
        return \
            sql.select([
                sql.distinct(unclassified_lookup).label('student_lookup'),
                sql.literal(original_labels_config.unconfirmed_low_gpa).label('label')
            ]).\
            where(
                unclassified_gpa_seniors.c.gpa < low_gpa_thresh.select()
            )

    def get_unconfirmed_high_gpa(self, unclassified_gpa_seniors, high_gpa_thresh):
        unclassified_lookup = unclassified_gpa_seniors.c.student_lookup
        return \
            sql.select([
                sql.distinct(unclassified_lookup).label('student_lookup'),
                sql.literal(original_labels_config.unconfirmed_high_gpa).label('label')
            ]).\
            where(
                unclassified_gpa_seniors.c.gpa > high_gpa_thresh.select()
            )

    def get_unclassified(self, all_snapshots, on_time_grads, late_grads, dropouts, unconfirmed_high_gpa, unconfirmed_low_gpa):
        student_lookup              = all_snapshots.c.student_lookup
        grade                       = all_snapshots.c.grade
        on_time_lookups             = sql.select([on_time_grads.c.student_lookup])
        late_lookups                = sql.select([late_grads.c.student_lookup])
        dropout_lookups             = sql.select([dropouts.c.student_lookup])
        unconfirmed_high_lookups    = sql.select([unconfirmed_high_gpa.c.student_lookup])
        unconfirmed_low_lookups     = sql.select([unconfirmed_low_gpa.c.student_lookup])

        u = sql.union(on_time_lookups, late_lookups, dropout_lookups, unconfirmed_high_lookups, unconfirmed_low_lookups)

        return\
            sql.select([
                sql.distinct(student_lookup).label('student_lookup'),
                sql.literal(original_labels_config.unclassified).label('label')
            ]).\
            where(
                db_expr.and_(
                    student_lookup.notin_(u),
                    grade >= 12
                )
            )
