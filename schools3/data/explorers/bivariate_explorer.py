import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from schools3.data.explorers.explorer import Explorer
from schools3.data import common_queries
from schools3.config import global_config
from schools3.config.data.explorers import bivariate_explorer_config as config


class BivariateExplorer(Explorer):
    def __init__(self, debug=False):
        super(BivariateExplorer, self).__init__(debug)
        self.bivariate_df = pd.DataFrame(columns=[
            # 'discipline_incidents_rate',
            'absenteeism_rate',
            'final_gpa',
            # 'extracurr_invs',
            'academic_invs'
        ])

    def explore(self):
        students_info = self.query(common_queries.get_student_data([9, 12]))

        # self.bivariate_df.discipline_incidents_rate = students_info.discipline_incidents_rate.fillna(0)
        self.bivariate_df.absenteeism_rate = students_info.absenteeism_rate
        self.bivariate_df.final_gpa = students_info.gpas.apply(
                                        lambda x: x if x is None else x[-1]
                                    ).astype(float)
        self.bivariate_df.academic_invs = students_info.inv_groups.apply(
                                            lambda x: 0 if x is None else x.count('academic_inv')
                                        )
        # self.bivariate_df.extracurr_invs = students_info.inv_groups.apply(
        #                                     lambda x: 0 if x is None else x.count('atheletics') + x.count('extracurr_program')
        #                                 )

        self.bivariate_df.label = students_info.label

        self.bivariate_df.dropna(inplace=True)

        plt.rcParams['axes.labelsize'] = 13
        fig = sns.pairplot(self.bivariate_df, kind='hist', height=3)
        path = global_config.get_save_path(config.pairplot_save_file)
        fig.savefig(path, bbox_inches='tight')
        return fig
