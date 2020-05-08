#!/usr/bin/env python
# coding: utf-8

# In[ ]:


done = ['090-noun-titles.jsonl', '006-entity-titles.jsonl', '108-nouns.jsonl', '027-nouns.jsonl', '040-nouns.jsonl', '015-entity-titles.jsonl', '102-entity-titles.jsonl', '100-noun-titles.jsonl', '015-noun-titles.jsonl', '078-entity-titles.jsonl', '100-entity-titles.jsonl', '081-entities.jsonl', '028-noun-titles.jsonl', '089-noun-titles.jsonl', '089-nouns.jsonl', ]


# In[1]:


import glob
import pandas as pd
import json
import datetime
import numpy as np


# In[2]:


from pymongo import MongoClient
client = MongoClient()
mdb = client.fever2


# In[171]:


df = pd.read_json(f'../data/spacy_v3/089-nouns.jsonl', orient='records', lines=True)


# In[166]:


df['_id']


# In[167]:


c = mdb['noun-titles'].find({'_id': {'$in': list(df['_id'])}})


# In[168]:


there = []
for i in c:
    there.append(i['_id'])


# In[169]:


len(there)


# In[170]:


res = df.to_dict('records')
mdb['noun-titles'].insert_many(res)


# In[77]:


df1 = df1[~df1['_id'].isin(there)]


# In[78]:


df1


# In[ ]:




