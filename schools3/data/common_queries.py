import sqlalchemy as sql
import sqlalchemy.sql.functions as db_func
from schools3.config.data import db_tables
from sqlalchemy.dialects.postgresql import aggregate_order_by


def get_student_data(grade_bounds):
    metadata = sql.MetaData()

    all_snapshots = db_tables.clean_all_snapshots_table
    hs_grade_gpa = get_students_grade_gpa().cte('hs_grade_gpa')
    inv_table = db_tables.clean_intervention_table

    get_ordered_array = lambda c, o : db_func.array_agg(aggregate_order_by(c, o))

    discipline_incidents_rate = \
        db_func.sum(all_snapshots.c.discipline_incidents) /\
            db_func.count(sql.distinct(all_snapshots.c.school_year))

    absenteeism_rate = db_func.sum(all_snapshots.c.days_absent) /\
            db_func.count(sql.distinct(all_snapshots.c.school_year))

    unexcused_absenteeism_rate = db_func.sum(all_snapshots.c.days_absent_unexcused) /\
        db_func.count(sql.distinct(all_snapshots.c.school_year))

    basic_info = sql.select([
                    all_snapshots.c.student_lookup,
                    db_func.max(all_snapshots.c.gender).label('gender'),
                    db_func.max(all_snapshots.c.ethnicity).label('ethnicity'),
                    discipline_incidents_rate.label('discipline_incidents_rate'),
                    absenteeism_rate.label('absenteeism_rate'),
                    unexcused_absenteeism_rate.label('unexcused_absenteeism_rate'),
                    db_func.array_agg(sql.distinct(all_snapshots.c.disability)).label('disabilities'),
                    db_func.array_agg(sql.distinct(all_snapshots.c.disadvantagement)).label('disadvantagements'),
                    db_func.array_agg(sql.distinct(all_snapshots.c.limited_english)).label('limited_english'),
                    db_func.array_agg(sql.distinct(all_snapshots.c.special_ed)).label('special_ed'),
                    db_func.max(all_snapshots.c.graduation_date).label('graduation_date'),
                    get_ordered_array(all_snapshots.c.school_code, all_snapshots.c.grade).label('school_codes'),
                    get_ordered_array(all_snapshots.c.school_name, all_snapshots.c.grade).label('school_names'),
                    get_ordered_array(all_snapshots.c.grade, all_snapshots.c.grade).label('snapshots_grades'),
                    get_ordered_array(all_snapshots.c.school_year, all_snapshots.c.grade).label('snapshots_school_years')
                ]).\
                where(
                    sql.and_(
                        all_snapshots.c.grade >= grade_bounds[0],
                        all_snapshots.c.grade <= grade_bounds[1]
                    )
                ).\
                group_by(
                    all_snapshots.c.student_lookup
                ).cte('basic_info')

    hs_gpa_info = sql.select([
                    hs_grade_gpa.c.student_lookup,
                    get_ordered_array(hs_grade_gpa.c.gpa, hs_grade_gpa.c.grade).label('gpas'),
                    get_ordered_array(hs_grade_gpa.c.grade, hs_grade_gpa.c.grade).label('hs_grades'),
                    get_ordered_array(hs_grade_gpa.c.school_year, hs_grade_gpa.c.grade).label('hs_school_years'),
                    get_ordered_array(hs_grade_gpa.c.num_classes, hs_grade_gpa.c.grade).label('num_classes')
                ]).where(
                    sql.and_(
                        hs_grade_gpa.c.grade >= grade_bounds[0],
                        hs_grade_gpa.c.grade <= grade_bounds[1]
                    )
                ).group_by(
                    hs_grade_gpa.c.student_lookup
                ).cte('hs_gpa_info')

    inv_info = sql.select([
                    inv_table.c.student_lookup,
                    get_ordered_array(inv_table.c.inv_group, inv_table.c.grade).label('inv_groups'),
                    get_ordered_array(inv_table.c.membership_code, inv_table.c.grade).label('membership_codes'),
                    get_ordered_array(inv_table.c.grade, inv_table.c.grade).label('inv_grades'),
                    get_ordered_array(inv_table.c.school_year, inv_table.c.grade).label('inv_school_years'),
                ]).where(
                    sql.and_(
                        inv_table.c.grade >= grade_bounds[0],
                        inv_table.c.grade <= grade_bounds[1]
                    )
                ).group_by(
                    inv_table.c.student_lookup
                ).cte('inv_info')

    labels = db_tables.sketch_temp_labels_table

    to_join = [basic_info, hs_gpa_info, inv_info, labels]

    joined = to_join[0]
    for i in range(1, len(to_join)):
        if i == 1:
            on_clause = (joined.c.student_lookup == to_join[i].c.student_lookup)
        else:
            on_clause = (joined.c[to_join[0].name +'_student_lookup'] == to_join[i].c.student_lookup)

        joined = sql.join(
                    left=joined, right=to_join[i],
                    onclause=on_clause,
                    isouter=True
                )

    cs = []
    added_student_lookup = False
    for c in joined.c:
        if c.name == 'student_lookup':
            if not added_student_lookup:
                cs.append(c)
                added_student_lookup = True
        else:
            cs.append(c)

    return sql.select(cs).select_from(joined)


def get_query_with_students(query, student_lookup_query):
    s = student_lookup_query.cte('s')
    student_lookups = sql.select([s.c.student_lookup]).cte('s_lookup')
    q = query.cte('query')

    joined = sql.join(
                student_lookups, q,
                onclause=(student_lookups.c.student_lookup == q.c.student_lookup),

            )

    return sql.select(
            [student_lookups.c.student_lookup] + [c for c in q.c if c.name != 'student_lookup']
        ).select_from(
            joined
        )


def get_students_grade_gpa():
    '''
        Returns a query that can returns a table with a "grade" column to the
        high_school_gpa table
    '''
    high_school_gpa = db_tables.clean_high_school_gpa_table
    all_snapshots = db_tables.clean_all_snapshots_table

    left = sql.select([
                    all_snapshots.c.student_lookup,
                    all_snapshots.c.grade,
                    all_snapshots.c.school_year
                ]).\
                where(
                    sql.and_(
                        all_snapshots.c.grade >= 9,
                        all_snapshots.c.grade <= 12
                    )
            ).alias('a')

    right = high_school_gpa.alias('b')

    joined = sql.join(
            left=left,
            right=right,
            onclause=sql.and_(
                    left.c.student_lookup == right.c.student_lookup,
                    left.c.school_year == right.c.school_year,
                )
            )

    return sql.select([
        joined.c.a_student_lookup,
        joined.c.a_grade,
        joined.c.a_school_year,
        joined.c.b_gpa,
        joined.c.b_num_classes
    ]).\
    select_from(joined).\
    group_by(*list(joined.c))


def get_snapshot_students(cols=[], hs_only=True):
    assert isinstance(cols, list), 'cols must be a list'
    all_snapshots = db_tables.clean_all_snapshots_table

    select_cols = [
        all_snapshots.c.student_lookup,
        all_snapshots.c.school_year,
        all_snapshots.c.grade
    ] + cols

    if hs_only:
        return sql.select(
                    select_cols
                ).where(
                    all_snapshots.c.grade >= 9
                )

    return sql.select(
                select_cols
            )

def get_labels():
    labels_table = db_tables.sketch_temp_labels_table

    return sql.select([sql.distinct(labels_table.c.label)])


def get_students_with_label(label):
    labels_table = db_tables.sketch_temp_labels_table

    return \
        sql.select(
            [labels_table.c.student_lookup]
        ).where(
            labels_table.c.label == label
        )
