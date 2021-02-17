import pandas as pd
from schools3.data.features.processors.feature_processor import FeatureProcessor
import schools3.config.data.features.acs_features_config as config


class ACSFeatureProcessor(FeatureProcessor):
    def get_base_category(self, name):
        '''
            helper function to simply get the base category of a variable name
        '''
        return '!!'.join(name.split('!!')[:2])

    def get_parent_category(self, name):
        '''
            helper function to simply get the parent prefix of a variable name
        '''
        return '!!'.join(name.split('!!')[:-1])

    def __call__(self, data):
        group_vars = config.get_group_variables()

        for g in group_vars:
            links = {}
            labels = {}
            smoothing_num = {name: 0 for _, name in group_vars[g].items()}

            sorted_col_names = \
                sorted(group_vars[g].items(), key=lambda x: -len(x[1].split('!!')))

            for k, name in sorted_col_names:
                base_cat = self.get_base_category(name)
                parent_cat = self.get_parent_category(name)

                labels[name] = k

                if parent_cat in smoothing_num:
                    if smoothing_num[name] == 0:
                        smoothing_num[name] += 1
                    smoothing_num[parent_cat] += smoothing_num[name]

                if name != base_cat:
                    links[name] = base_cat

            for k in sorted(links, key=lambda x: -len(x.split('!!'))):
                base_cat = links[k]
                if labels[base_cat] in data.columns:
                    smoothed_numer = (data[labels[k]] + smoothing_num[k])
                    smoothed_denom = (data[labels[base_cat]] + smoothing_num[base_cat])
 
                    data[labels[k]] = smoothed_numer / smoothed_denom

        return data
