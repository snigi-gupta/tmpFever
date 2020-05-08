#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autotime')


# In[2]:


import pandas as pd
import spacy
import numpy as np
import json


# In[5]:


nlp = spacy.load('en_core_web_sm')


# In[6]:


def find_entities(s):
    doc = nlp(s)
    ents = doc.ents
    result = []
    for ent in ents:
        result.append([ent.label_, str(ent.text)])
    if len(result) == 0:
        result = [[]]
    return result


# In[7]:


def build(fname):
    df = pd.read_json(f'../data/wiki-pages/wiki-{fname}.jsonl', orient='records', lines=True)
    df['text'] = df['text'].replace('', np.nan)
    df.dropna(subset=['text'], inplace=True)
    entities = df['text'].apply(lambda x: find_entities(x))
    entities_values = entities.explode()
    entities_index = entities_values.index
    entities_df = pd.DataFrame(list(entities_values))
    entities_df['doc_id'] = entities_index
    j = entities_df.to_json(orient='records', lines=True)
    f = open(f'../data/doc_ret/{fname}.jsonl', 'w+')
    f.write(json.dumps(j))
    f.close()
    print(f'done {fname}')


# In[8]:


docs = ['025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039']


# In[9]:


for doc in docs:
    build(doc)


# In[ ]:




