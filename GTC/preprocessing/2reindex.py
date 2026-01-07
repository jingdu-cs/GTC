import os
import pandas as pd

i_id_mapping = '../data/cell/i_id_mapping.csv'
df = pd.read_csv(i_id_mapping, sep='\t')
print(f'shape: {df.shape}')

import gzip, json
meta_file = '../data/cell/meta_Cell_Phones_and_Accessories.json.gz'

print('0 Extracting U-I interactions.')

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

meta_df = getDF(meta_file)

print(f'Total records: {meta_df.shape}')


map_dict = dict(zip(df['asin'], df['itemID']))

meta_df['itemID'] = meta_df['asin'].map(map_dict)
meta_df.dropna(subset=['itemID'], inplace=True)
meta_df['itemID'] = meta_df['itemID'].astype('int64')
#meta_df['description'] = meta_df['description'].fillna(" ")
meta_df.sort_values(by=['itemID'], inplace=True)

print(f'shape: {meta_df.shape}')

ori_cols = meta_df.columns.tolist()

ret_cols = [ori_cols[-1]] + ori_cols[:-1]
print(f'new column names: {ret_cols}')

ret_df = meta_df[ret_cols]
# dump
ret_df.to_csv(os.path.join('./', '../data/cell/meta-cell14.csv'), index=False)
print('done!')

indexed_df = pd.read_csv('../data/cell/meta-cell14.csv')
print(f'shape: {indexed_df.shape}')

i_uni = indexed_df['itemID'].unique()

print(f'# of unique items: {len(i_uni)}')

print('min/max of unique learners: {0}/{1}'.format(min(i_uni), max(i_uni)))