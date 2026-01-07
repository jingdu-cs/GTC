import os
import numpy as np
import pandas as pd

i_id, desc_str = 'itemID', 'description'

file_path = '../data/cell/'
file_name = 'meta-cell14.csv'

meta_file = os.path.join(file_path, file_name)

df = pd.read_csv(meta_file)
df.sort_values(by=[i_id], inplace=True)

print('data loaded!')
print(f'shape: {df.shape}')

title_na_df = df[df['title'].isnull()]
print(title_na_df.shape)

desc_na_df = df[df['description'].isnull()]
print(desc_na_df.shape)

na_df = df[df['description'].isnull() & df['title'].isnull()]
print(na_df.shape)

na3_df = df[df['description'].isnull() & df['title'].isnull() & df['brand'].isnull()]
print(na3_df.shape)

na4_df = df[df['description'].isnull() & df['title'].isnull() & df['brand'].isnull() & df['categories'].isnull()]
print(na4_df.shape)

df[desc_str] = df[desc_str].fillna(" ")
df['title'] = df['title'].fillna(" ")
df['brand'] = df['brand'].fillna(" ")
df['categories'] = df['categories'].fillna(" ")

sentences = []
for i, row in df.iterrows():
    sen = row['title'] + ' ' + row['brand'] + ' '
    cates = eval(row['categories'])
    if isinstance(cates, list):
        for c in cates[0]:
            sen = sen + c + ' '
    sen += row[desc_str]
    sen = sen.replace('\n', ' ')

    sentences.append(sen)

course_list = df[i_id].tolist()
#sentences = df[desc_str].tolist()

assert course_list[-1] == len(course_list) - 1

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence_embeddings = model.encode(sentences)
print('text encoded!')

assert sentence_embeddings.shape[0] == df.shape[0]
np.save(os.path.join(file_path, 'text_feat.npy'), sentence_embeddings)
print('done!')

load_txt_feat = np.load('../data/cell/text_feat.npy', allow_pickle=True)
print(load_txt_feat.shape)

import array

def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10).decode('UTF-8')
    if asin == '': break
    a = array.array('f')
    a.fromfile(f, 4096)
    yield asin, a.tolist()

img_data = readImageFeatures("../data/cell/image_features_Cell_Phones_and_Accessories.b")
item2id = dict(zip(df['asin'], df['itemID']))

feats = {}
avg = []
for d in img_data:
    if d[0] in item2id:
        feats[int(item2id[d[0]])] = d[1]
        avg.append(d[1])
avg = np.array(avg).mean(0).tolist()

ret = []
non_no = []
for i in range(len(item2id)):
    if i in feats:
        ret.append(feats[i])
    else:
        non_no.append(i)
        ret.append(avg)

print('# of items not in processed image features:', len(non_no))
assert len(ret) == len(item2id)
np.save('../data/cell/image_feat.npy', np.array(ret))
np.savetxt("missed_img_itemIDs.csv", non_no, delimiter =",", fmt ='%d')
print('done!')