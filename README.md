# OH Schools Team 3

## Instructions to run

Please put your username in this [file](./schools3/config/data/db_config.py#L10).

```bash
source /data/groups/schools3/dssg_env/bin/activate  # can add this in ~/.bashrc file
python3 schools3/main.py --exp SingleDatasetExperiment
```

In the call to `main.py`, we can swap out the `--exp` argument with any class in the [experiments](https://github.com/dssg/mlpolicylab_fall20_schools3/tree/master/schools3/ml/experiments) directory to run different workflows. We support the following options for this argument:
- `SingleDatasetExperiment`: trains models and reports metrics for a single grade
- `MultiDatasetExperiment`: trains models and reports metrics for multiple grades 
- `LocalImportancesExperiment`: finds locally important features for models using LIME
- `HPTuningExperiment`: does hyperparameter tuning
- `FeatureImportancesExperiment`: find most important features for models using permutation importance
- `FeaturePruningExperiment`: gets feature importances, then trains models and report metrics when only given some number of most important features
- `CrossTabsExperiment`: finds the features with greatest disparity between the positive and negative labels

The other (optional) arguments for main are
- `--no-cache`: if not given, models may be read from a cache instead of being re-trained
- `--no-good-values`: used for in `HPTuningExperiment`. If not given, the hyperparameter tuning library will only iterate over grid values that we specified in the config. 

The settings for all experiments are controlled by the [main config](./schools3/config/main_config.py). From this file, we can control
- years
- grade(s)
- features and feature processors
- labels
- metrics
- fairness attributes
- caching behavior
- hyperparameter tuning strategy

Similar to the experiments, most of these settings (such as features and models) have corresponding directory and are selected by just specifying a class. The Repository Structure section of the README gives an overview of all of the options available.

## Repository Structure

Below is the tree structure of the notable files in our repo. Descriptions of the directories are in bold

Note that the `config` and `gen` directories have internal directory structure that mirrors the schools3 directory. This means that a file `schools3/foo/bar/file.py` would have its corresponding configuration placed at `schools3/config/foo/bar/file_config.py`.

<pre>
schools3
├── main.py
├── config - <b>where config variables are stored. The structure is a mirror of the schools3 directory</b>
│   ├── data
│   │   ├── db_config.py
│   │   └── db_tables.py
│   ├── base_config.py
│   ├── global_config.py
│   └── main_config.py
├── data - <b>data for machine learning</b>
│   ├── acs
│   │   └── acs_table.py
│   ├── base - <b>base classes for the data/ dir</b>
│   │   ├── cohort.py
│   │   ├── cohortable_table.py
│   │   ├── processor.py
│   │   └── schools_table.py
│   ├── datasets - <b>input class for a model. It wraps around a cohort, features, and labels</b>
│   │   ├── dataset.py
│   │   └── datasets_generator.py
│   ├── explorers - <b>code to do data exploration for proposal</b>
│   │   ├── bivariate_explorer.py
│   │   ├── explorer.py
│   │   ├── labels_explorer.py
│   │   └── temporal_explorer.py
│   ├── features - <b>tables of features</b>
│   │   ├── processors - <b>classes that can transform the tables of features</b>
│   │   │   ├── acs_feature_processor.py
│   │   │   ├── categorical_feature_processor.py
│   │   │   ├── composite_feature_processor.py
│   │   │   ├── feature_processor.py
│   │   │   ├── impute_null_processor.py
│   │   │   ├── inv_type_processor.py
│   │   │   ├── oaaogt_processor.py
│   │   │   ├── pivot_processor.py
│   │   │   ├── replace_nullish_processor.py
│   │   │   ├── standardize_processor.py
│   │   │   └── transform_processor.py
│   │   ├── absence_desc_features.py
│   │   ├── absence_features.py
│   │   ├── academic_features.py
│   │   ├── acs_features.py
│   │   ├── discipline_incident_rate_features.py
│   │   ├── features_table.py
│   │   ├── inv_features.py
│   │   ├── inv_type_features.py
│   │   ├── middle_school_features.py
│   │   ├── oaaogt_features.py
│   │   └── pivot_block_features.py
│   └── labels - <b>tables of labels</b>
│       ├── labels_table.py
│       └── original_labels.py
├── gen - <b>generated output. The structure is a mirror of the schools3 dir</b>
└── ml - <b>everything related to learning</b>
    ├── base - <b>base classes for the ml/ dir</b>
    │   ├── experiment.py
    │   ├── hyperparameters.py
    │   ├── metrics.py
    │   └── model.py
    ├── experiments - <b>different top-level tasks to be called from main</b>
    │   ├── cross_tabs_experiment.py
    │   ├── feat_importances_experiment.py
    │   ├── feat_pruning_experiment.py
    │   ├── hp_tuning_experiment.py
    │   ├── local_importances_experiment.py
    │   ├── models_experiment.py
    │   ├── multi_dataset_experiment.py
    │   └── single_dataset_experiment.py
    ├── metrics - <b>metrics to evaluate a model with</b>
    │   ├── composite_metrics.py
    │   ├── fairness_metrics.py
    │   ├── performance_metrics.py
    │   └── prk_curve_metrics.py
    ├── baselines - <b>baselines to compare the machine learning model against</b>
    │   ├── absenteeism_baseline.py
    │   ├── baseline.py
    │   ├── discipline_baseline.py
    │   ├── gpa_baseline.py
    │   └── ranked_baseline.py
    ├── hyperparameters - <b>hyperparameter definitions for machine learning model</b>
    │   ├── decision_tree_hyperparameters.py
    │   ├── gradient_boosting_hyperparameters.py
    │   ├── k_neighbors_hyperparameters.py
    │   ├── logistic_regression_hyperparameters.py
    │   ├── mlp_hyperparameters.py
    │   ├── random_forest_hyperparameters.py
    │   └── svm_hyperparameters.py
    └── models - <b>the machine learning models</b>
        ├── all_ones_model.py
        ├── decision_tree_model.py
        ├── ensemble_model.py
        ├── gradient_boosting_model.py
        ├── k_neighbors_model.py
        ├── logistic_regression_model.py
        ├── mlp_model.py
        ├── random_forest_model.py
        ├── sklearn_model.py
        ├── svm_model.py
        └── tfkeras_model.py
</pre>

## OH School Data Notes

According to the US Department of Education, 1 in 5 high school students fail to graduate within four years, but at risk students are typically only identified late in high school after significant red flags. This project works with data from the Muskingum Valley Education Service Center (MVESC) with the goal of supplementing educators' intuitions to provider an earlier warning about which students are in need of targeted interventions. MVESC provides services to 16 school districts in 5 counties in rural southeastern Ohio, reaching more than 30,000 pre-K to high school students annually.

MVESC has provided longitudinal data for seven of the districts it works with dating back to the 2006-2007 school year (except Ridgewood county, which starts in 2007), including information about grades, absences, standardized tests, and withdrawl information. Raw data was provided as a set of SQL Server backups, CSV files, and Exel files and has been ingested into a postgres database. These raw data can be found in several tables in the `public` schema, however, initial cleaning, consolidation, and recoding has been performed to generate the tables in the `clean` schema with a student-level identifier, `student_lookup`. These include:
- **Grades**: Information about courses and grades is available in `clean.all_grades` and derived from the underlying raw tables with "grade" in the table name. Note that the course grades are in the `mark` column while the `grade` column indicates the grade level of the student. These data have also been aggregated to an overall GPA for high school students in `clean.high_school_gpa` on a 4-point scale.
- **Absences**: Data about absences and tardiness is found in `clean.all_absences` and derived from underlying raw tables with "absence" in the table name. Note that the `absence_code` is not standardized across schools and the `absence_desc` has been consolidated from the underlying data.
- **Test Scores**: Scores from standardized tests --- including Ohio Achievement Assessment (OAA), Ohio Graduation Test (OGT), Kindergarten Readiness Assessment for Literacy (KRA-L) --- are provided in `clean.oaaogt`. The OAA results are broken out by grade level and subject area and for all tests listed, columns ending with `_ss` indicate a numeric score and those ending with `_pl` indicate a placement level. On the OGT, scores of 400 or higher are considered passing in each subject area and scores fall into five [performance levels](http://ogt.success-ode-state-oh-us.info/resources/levelsMean.htm).
- **Interventions**: The `clean.interventions` table describes students who have received various types of interventions (e.g., special academic programs or instruction, extracurricular programs, etc.) in a given year. The underlying data tables are those beginning with `INV`. Notes that the "membership codes" relate to the program description --- a lookup can be found in `public."INV_MembershipCodes"`.
- **Student Information**: The `clean.all_snapshots` table contains student-level information about demographics, special needs, discipline (including in-school suspension `iss` and out-of-school suspension `oss`), etc. In this table, `section_504_plan` referrs to accomadations for students with disabilities under Section 504 of the Rehabilitation Act of 1973, and `IRN` referrs to an "Information Retrevial Number" that is used by the Ohio Department of Education to identify entities that interact with it (schools, districts, etc) which can be searched [here](https://oeds.ode.state.oh.us/SearchOrg).

## OH School Additional Data Information

[This](https://docs.google.com/document/d/1zoRQY7EHnOM5473hNJeLZPF6AazOw0CMiFrVNqxlyqk/edit) Google document contains some preliminary information about data and some known problems. Please add any new insights/problems that you find out in the document, so the team is updated.
