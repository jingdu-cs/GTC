import os, csv
import pandas as pd

rslt_file = '../data/cell/cell-indexed.inter'
df = pd.read_csv(rslt_file, sep='\t')
print(f'shape: {df.shape}')

import random
import numpy as np

df = df.sample(frac=1).reset_index(drop=True)

df.sort_values(by=['userID'], inplace=True)

uid_field, iid_field = 'userID', 'itemID'

uid_freq = df.groupby(uid_field)[iid_field]
u_i_dict = {}
for u, u_ls in uid_freq:
    u_i_dict[u] = list(u_ls)

new_label = []
u_ids_sorted = sorted(u_i_dict.keys())

for u in u_ids_sorted:
    items = u_i_dict[u]
    n_items = len(items)
    if n_items < 10:
        tmp_ls = [0] * (n_items - 2) + [1] + [2]
    else:
        val_test_len = int(n_items * 0.2)
        train_len = n_items - val_test_len
        val_len = val_test_len // 2
        test_len = val_test_len - val_len
        tmp_ls = [0] * train_len + [1] * val_len + [2] * test_len
    new_label.extend(tmp_ls)

df['x_label'] = new_label
new_labeled_file = rslt_file[:-6] + '-v4.inter'
df.to_csv(os.path.join('./', new_labeled_file), sep='\t', index=False)
print('done!!!')