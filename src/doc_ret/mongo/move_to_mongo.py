#!/usr/bin/env python
# coding: utf-8

# In[17]:


import glob
import pandas as pd
import json
import datetime
import numpy as np


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autotime')


# In[20]:


files = ['006-noun-titles.jsonl', '093-nouns.jsonl', '006-entities.jsonl',
        '106-nouns.jsonl', '087-entities.jsonl', '017-noun-titles.jsonl',
        '040-noun-titles.jsonl', '089-entity-titles.jsonl',
        '096-entities.jsonl', '027-noun-titles.jsonl',
        '102-entities.jsonl', '081-entity-titles.jsonl',
        '087-noun-titles.jsonl', '058-entity-titles.jsonl',
        '106-entity-titles.jsonl', '058-noun-titles.jsonl',
        '031-nouns.jsonl', '096-entity-titles.jsonl',
        '020-noun-titles.jsonl', '006-nouns.jsonl', '089-entities.jsonl',
        '015-nouns.jsonl', '060-entities.jsonl', '060-entity-titles.jsonl',
        '080-nouns.jsonl', '083-noun-titles.jsonl', '078-nouns.jsonl',
        '027-entities.jsonl', '090-entity-titles.jsonl',
        '100-entities.jsonl', '108-entity-titles.jsonl',
        '040-entity-titles.jsonl', '091-entity-titles.jsonl',
        '002-entity-titles.jsonl', '034-entities.jsonl',
        '093-entity-titles.jsonl', '108-noun-titles.jsonl',
        '085-entities.jsonl', '080-noun-titles.jsonl', '003-nouns.jsonl',
        '091-noun-titles.jsonl', '081-nouns.jsonl', '087-nouns.jsonl',
        '080-entities.jsonl', '058-nouns.jsonl', '028-nouns.jsonl',
        '017-entity-titles.jsonl', '028-entities.jsonl', '083-nouns.jsonl',
        '091-nouns.jsonl', '003-noun-titles.jsonl',
        '003-entity-titles.jsonl', '093-noun-titles.jsonl']


# In[21]:


from pymongo import MongoClient
client = MongoClient()
mdb = client.fever2


# In[22]:


def get_collection_name(fname):
    return "_".join(fname.split('.')[0].split('-')[1:])

def write_to_db(db, data):
    total = len(data)
    batch_size = 20000
    i = 0
    while total > 0:
        y = i * batch_size
        to_insert = data[y: y + batch_size]
        db.insert_many(to_insert, ordered=False)
        total -= batch_size
        i += 1


# In[24]:


for fname in files[1:]:
    s = datetime.datetime.now()
    df = pd.read_json(f'../data/spacy_v3/{fname}', orient='records', lines=True)
    collection = get_collection_name(fname)
    res = df.to_dict('records')
    mdb[collection].insert_many(res)
    print(f'done {fname} - {datetime.datetime.now() - s}')


# In[ ]:


done = ['090-noun-titles.jsonl', '006-entity-titles.jsonl', '108-nouns.jsonl', '027-nouns.jsonl', '040-nouns.jsonl', '015-entity-titles.jsonl', '102-entity-titles.jsonl', '100-noun-titles.jsonl', '015-noun-titles.jsonl', '078-entity-titles.jsonl', '100-entity-titles.jsonl', '081-entities.jsonl', '028-noun-titles.jsonl', '089-noun-titles.jsonl', '089-nouns.jsonl', '006-noun-titles.jsonl']


# In[18]:


np.array_split(files, 2)


# In[26]:


df['_id'][0]


# In[ ]:




