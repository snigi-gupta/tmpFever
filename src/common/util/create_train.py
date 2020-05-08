#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import glob
import json
import unidecode


# In[2]:


get_ipython().run_line_magic('load_ext', 'autotime')


# In[3]:


wiki_pages = glob.glob('data/wiki-pages/*')


# In[4]:


df = pd.DataFrame()


# In[5]:


devdf = pd.read_json('data/fever-data/train.jsonl', lines=True, orient='records')
devdf = devdf[devdf['verifiable'] == 'VERIFIABLE']


# In[6]:


devdf


# In[7]:


values = devdf['evidence'].explode().explode()
index = values.index


# In[8]:


train = pd.DataFrame(list(values))
train['claim'] = devdf['claim'][index].reset_index()['claim']
train['id'] = devdf['id'][index].reset_index()['id']
train['label'] = devdf['label'][index].reset_index()['label']
del train[0]
del train[1]

train.columns = ['id', 'sentence', 'claim', 'claim_id', 'label']


# In[11]:


train['match_id'] = train['id'].apply(lambda x: unidecode.unidecode(x))


# In[12]:


train['match_id'][219683]


# In[13]:


def split_into_id_and_lines(s):
    lines = s.split('\n')
    result = []
    for line in lines:
        sentence_id, text = line.split('\t', 1)
        result.append([sentence_id, text])
    return result


def merge_dfs(pages, wiki_dump, train, i):
    tmp = wiki_dump[wiki_dump['match_id'].isin(pages)]
    tmp['lines'] = tmp['lines'].apply(lambda x: split_into_id_and_lines(x))
    train = train.merge(tmp, how='left', left_on='match_id', right_on='match_id')
    if i > 0:
        train['id'] = train['id_x'].fillna(train['id_y'])
        train['text'] = train['text_x'].fillna(train['text_y'])
        train['lines'] = train['lines_x'].fillna(train['lines_y'])
        train = train.drop(labels=['text_x', 'text_y', 'lines_x', 'lines_y', 'id_x', 'id_y'], axis=1)
    return train


# In[14]:


batch_size = 10
wiki_pages_chunks = np.array_split(wiki_pages, batch_size)
i = 0
for chunks in wiki_pages_chunks:
    df = pd.DataFrame()
    for f in chunks:
        d = pd.read_json(f, orient='records', lines=True)
        df = df.append(d, ignore_index=True)
        print(f'read {f}')
    del d
    df['match_id'] = df['id'].apply(lambda x: unidecode.unidecode(x))
    train = merge_dfs(train['match_id'], df, train, i)
    print(f'done chunk {i}')
    i += 1


# In[23]:


del df


# In[16]:


train


# In[17]:


train.isnull().values.any()


# In[22]:


train[train['match_id'] == 'Bjorn_Ulvaeus']

