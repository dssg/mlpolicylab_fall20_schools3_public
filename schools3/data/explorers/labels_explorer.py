import pandas as pd
from schools3.data.explorers.explorer import Explorer
from schools3.data import common_queries


class LabelsExplorer(Explorer):
    def __init__(self, debug=False):
        super(LabelsExplorer, self).__init__(debug)
        self.labels_df = pd.DataFrame(columns=[
            'label',
            'num_students',
            'gender_dist',
            'ethnicity_dist',
            'discipline_incidents_rate',
            'absenteeism_rate',
            'unexcused_absenteeism_rate',
            'disabilities_dist',
            'num_disabilities',
            'disadvantagements_dist',
            'latest_gpa',
            'total_classes',
            'inv_groups_dist',
            'avg_academic_inv',
            'avg_extracurr_inv'
        ])

    def explore(self):
        all_students_info_q = common_queries.get_student_data([9, 12])
        labels = self.query(common_queries.get_labels())

        for l in labels['label']:
            students_q = common_queries.get_students_with_label(l)
            cur_students_info = self.query(
                                    common_queries.get_query_with_students(
                                        all_students_info_q,
                                        students_q
                                    )
                                )
            self.labels_df = self.append_label_info(cur_students_info, self.labels_df, l)

        return self.labels_df

    def append_label_info(self, cur_students_info, labels_df, label):
        labels_df = labels_df.append(pd.Series(), ignore_index=True)

        new_row = labels_df.iloc[-1]

        new_row.label = label
        new_row.num_students = cur_students_info.shape[0]

        new_row.gender_dist = self.get_dist(cur_students_info.gender)
        new_row.ethnicity_dist = self.get_dist(cur_students_info.ethnicity)

        new_row.discipline_incidents_rate = cur_students_info.discipline_incidents_rate.fillna(0).mean()
        new_row.absenteeism_rate = cur_students_info.absenteeism_rate.mean()
        new_row.unexcused_absenteeism_rate = cur_students_info.unexcused_absenteeism_rate.mean()

        new_row.disabilities_dist = self.get_dist(cur_students_info.disabilities, is_element_list=True, normalize=False)
        new_row.num_disabilities = cur_students_info.disabilities.apply(
                                        lambda x : len([e for e in x if e and e != 'none'])
                                    ).mean()
        new_row.disadvantagements_dist = self.get_dist(cur_students_info.disadvantagements, is_element_list=True, normalize=False)

        new_row.latest_gpa = cur_students_info.gpas.apply(
                                    lambda x : x if x is None else x[-1]
                                ).mean()
        new_row.total_classes = cur_students_info.num_classes.apply(
                                    lambda x : x if x is None else sum(x)
                                ).mean()

        new_row.inv_groups_dist = self.get_dist(
                                    cur_students_info.inv_groups,
                                    is_element_list=True,
                                    normalize=False
                                )

        new_row.avg_academic_inv = cur_students_info.inv_groups.apply(
                                            lambda x : 0 if not x else x.count('academic_inv')
                                        ).mean()

        new_row.avg_extracurr_inv = cur_students_info.inv_groups.apply(
                                             lambda x: 0 if x is None else x.count('atheletics') + x.count('extracurr_program')
                                        ).mean()

        return labels_df

    def get_dist(self, series, is_element_list=False, normalize=True, normalize_by=None):
        if not is_element_list:
            if normalize_by is None:
                d = series.value_counts(normalize=normalize).to_dict()
            elif normalize:
                d = series.value_counts().to_dict()
                for k in d:
                    d[k] /= normalize_by
            return d

        d = {}
        for s in series:
            if s:
                for e in s:
                    if e and e.lower() != 'none':
                        if e in d:
                            d[e] += 1
                        else:
                            d[e] = 1

        if normalize:
            total = sum(d.values()) if not normalize_by else normalize_by
            for k in d:
                d[k] /= total

        return d
