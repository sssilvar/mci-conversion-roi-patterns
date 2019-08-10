from os.path import join, dirname, realpath

import pandas as pd

root = dirname(dirname(realpath(__file__)))
adnimerge_csv = join(root, 'data/df_conversions_with_times.csv')
adnimerge = pd.read_csv(adnimerge_csv, index_col='PTID', low_memory=False)

# Replace dots for underscore in column names
cols = [i.replace('.', '_').upper() for i in adnimerge.columns]
adnimerge.columns = cols


def load_adnimerge():
    return adnimerge


def get_subject_label(subject_id):
    return adnimerge.loc[subject_id, 'TARGET']


def filter_adni_conv_time(conversion_time):
    # Filter by conversion/stability time
    query = f'MONTH_STABLE >= {conversion_time} or MONTH_CONVERSION <= {conversion_time}'
    return adnimerge.query(query)
