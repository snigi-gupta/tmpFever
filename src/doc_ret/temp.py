import pandas as pd
import numpy as np
import glob

wiki_pages = glob.glob('data/wiki-pages/*')

df = pd.DataFrame()

for f in wiki_pages:
    d = pd.read_json(f, orient='records', lines=True)
    del d['lines']
    del d['text']
    df = df.append(d, ignore_index=True)
    print(f'done {f}')

devdf = pd.read_json('data/fever-data/train.jsonl', lines=True, orient='records')
devdf = devdf[devdf['verifiable'] == 'VERIFIABLE']

values = devdf['evidence'].explode().explode()
index = values.index
df1 = pd.DataFrame(list(values))
df1['claim'] = devdf['claim'][index].reset_index()['claim']
df1['id'] = devdf['id'][index].reset_index()['id']
df1['label'] = devdf['label'][index].reset_index()['label']

df3 = df[df['id'].isin(df1[2])]
df1 = df1.merge(df3, how='left', left_on=2, right_on='id')

