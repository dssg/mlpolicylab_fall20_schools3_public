import sys
import numpy as np
import requests
from schools3.config import base_config
from schools3.data.features.processors.impute_null_processor import ImputeBy

config = base_config.Config()

config.categorical_columns = []


# important params
year                   = 2018

config.state_fips      = 39
config.for_param       = 'block group:*'
config.group_name_desc = {
    'B11016': 'HOUSEHOLD TYPE BY HOUSEHOLD SIZE',
    'B15003': 'EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER',
    'B23025': 'EMPLOYMENT STATUS FOR THE POPULATION 16 YEARS AND OVER'
}

# zipcode extractor params
config.zipcodes_fname = 'schools3/data/ZIP_TRACT_122016.csv'
config.zipcodes_dtypes =  {
    'ZIP': str, 
    'TRACT': str,
    'RES_RATIO': float,
    'BUS_RATIO': float,
    'OTH_RATIO': float,
    'TOT_RATIO': float,
}

# acs extractor params
config.key     = '184fc9798e379fa4bb145284a3c6f3f8e5ff7fb2'
config.headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:80.0) Gecko/20100101 Firefox/80.0',
}

config.get_in_param = lambda state, county: f'state:{state} county:{county}'

def get_counties():
    base_url = f'https://api.census.gov/data/{year}/acs/acs5'
    params = {
        'get': 'NAME',
        'for': 'county:*',
        'in' : f'state:{config.state_fips}',
        'key': config.key
    }

    counties = np.array(
        requests.get(base_url, params=params).json()[1:]
    )

    return counties[:, 2], counties[:, 0]

config.get_counties = get_counties

config.all_variables_link  = f'https://api.census.gov/data/{year}/acs/acs5/variables.json'
config.groups_link         = f'https://api.census.gov/data/{year}/acs/acs5/groups.json'
config.get_group_vars_link = lambda group: f'https://api.census.gov/data/{year}/acs/acs5/groups/{group}.json'
config.raw_data_base_link  = f'https://api.census.gov/data/{year}/acs/acs5'

def get_group_variables(group_name_desc=config.group_name_desc):
    '''
        given group variables, the function returns names of all individual
        estimate variables which will be used to query APIs later
    '''
    if hasattr(config, 'group_variables'):
        return config.group_variables

    groups_link = config.groups_link
    groups = requests.get(groups_link).json()['groups']

    group_vars = {}
    for d in groups:
        if d['name'] in group_name_desc:
            # verify that group_name and group_desc match
            assert \
                d['description'].lower() == group_name_desc[d['name']].lower(),\
                f'{d["name"]} and {group_name_desc[d["name"]]} do not match'

            group_vars_link = config.get_group_vars_link(d['name'])
            fields = requests.get(group_vars_link).json()['variables']

            estimate_fields = {}
            for f in fields:
                if f.endswith('E'): # if field is an "estimate"
                    estimate_fields[f] = fields[f]['label']

            group_vars[d['name']] = estimate_fields
    
    config.group_variables = group_vars
    return group_vars

config.get_group_variables = get_group_variables

# sql params
config.schema_name = 'acs'

sys.modules[__name__] = config
