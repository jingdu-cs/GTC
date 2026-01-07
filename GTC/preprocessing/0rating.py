import os, csv
import pandas as pd

df = pd.read_csv('../data/cell/ratings_Cell_Phones_and_Accessories.csv', names=['userID', 'itemID', 'rating', 'timestamp'], header=None)
k_core = 5
learner_id, course_id, tmstmp_str = 'userID', 'itemID', 'timestamp'

df.dropna(subset=[learner_id, course_id, tmstmp_str], inplace=True)
df.drop_duplicates(subset=[learner_id, course_id, tmstmp_str], inplace=True)
print(f'After dropped: {df.shape}')

from collections import Counter
import numpy as np

min_u_num, min_i_num = 5, 5

def get_illegal_ids_by_inter_num(df, field, max_num=None, min_num=None):
    if field is None:
        return set()
    if max_num is None and min_num is None:
        return set()

    max_num = max_num or np.inf
    min_num = min_num or -1

    ids = df[field].values
    inter_num = Counter(ids)
    ids = {id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num}
    print(f'{len(ids)} illegal_ids_by_inter_num, field={field}')

    return ids


def filter_by_k_core(df):
    while True:
        ban_users = get_illegal_ids_by_inter_num(df, field=learner_id, max_num=None, min_num=min_u_num)
        ban_items = get_illegal_ids_by_inter_num(df, field=course_id, max_num=None, min_num=min_i_num)
        if len(ban_users) == 0 and len(ban_items) == 0:
            return

        dropped_inter = pd.Series(False, index=df.index)
        if learner_id:
            dropped_inter |= df[learner_id].isin(ban_users)
        if course_id:
            dropped_inter |= df[course_id].isin(ban_items)
        print(f'{len(dropped_inter)} dropped interactions')
        df.drop(df.index[dropped_inter], inplace=True)


filter_by_k_core(df)
print(f'k-core shape: {df.shape}')
print(f'shape after k-core: {df.shape}')
df.reset_index(drop=True, inplace=True)
i_mapping_file = '../data/cell/i_id_mapping.csv'
u_mapping_file = '../data/cell/u_id_mapping.csv'

splitting = [0.8, 0.1, 0.1]
uid_field, iid_field = learner_id, course_id

uni_users = pd.unique(df[uid_field])
uni_items = pd.unique(df[iid_field])

# start from 0
u_id_map = {k: i for i, k in enumerate(uni_users)}
i_id_map = {k: i for i, k in enumerate(uni_items)}

df[uid_field] = df[uid_field].map(u_id_map)
df[iid_field] = df[iid_field].map(i_id_map)
df[uid_field] = df[uid_field].astype(int)
df[iid_field] = df[iid_field].astype(int)

# dump
rslt_dir = './'
u_df = pd.DataFrame(list(u_id_map.items()), columns=['user_id', 'userID'])
i_df = pd.DataFrame(list(i_id_map.items()), columns=['asin', 'itemID'])

u_df.to_csv(os.path.join(rslt_dir, u_mapping_file), sep='\t', index=False)
i_df.to_csv(os.path.join(rslt_dir, i_mapping_file), sep='\t', index=False)
print(f'mapping dumped...')

# =========2. splitting
print(f'splitting ...')
tot_ratio = sum(splitting)
# remove 0.0 in ratios
ratios = [i for i in splitting if i > .0]
ratios = [_ / tot_ratio for _ in ratios]
split_ratios = np.cumsum(ratios)[:-1]

ts_id = 'timestamp'

split_timestamps = list(np.quantile(df[ts_id], split_ratios))
# get df training dataset unique users/items
df_train = df.loc[df[ts_id] < split_timestamps[0]].copy()
df_val = df.loc[(split_timestamps[0] <= df[ts_id]) & (df[ts_id] < split_timestamps[1])].copy()
df_test = df.loc[(split_timestamps[1] <= df[ts_id])].copy()

x_label, rslt_file = 'x_label', 'cell-indexed.inter'
df_train[x_label] = 0
df_val[x_label] = 1
df_test[x_label] = 2
temp_df = pd.concat([df_train, df_val, df_test])
temp_df = temp_df[[learner_id, course_id, 'rating', ts_id, x_label]]
print(f'columns: {temp_df.columns}')

temp_df.columns = [learner_id, course_id, 'rating', ts_id, x_label]

temp_df.to_csv(os.path.join(rslt_dir, rslt_file), sep='\t', index=False)