#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from db import db
import numpy as np
import json
import re
import spacy
from spacy.pipeline import EntityRuler


# In[2]:


get_ipython().run_line_magic('load_ext', 'autotime')


# In[77]:


woa = re.compile("(?<![a-z])([A-Z0-9].*) is a ([^\\.^\\n^\\t]*) (\\b(?:game|film|book|comic|movie|anime|comic)\\b)(?!(.creator|.director|.screenwriter|.creative|.writer|.actress|.actor))")


# In[78]:


def clean(s):
    s = s.replace('-LRB-', '(')
    s = s.replace('-LSB-', '[')
    s = s.replace('-RRB-', ')')
    s = s.replace('-RSB-', ']')
    s = s.replace('_', ' ')
    s = s.replace('-COLON-', ':')
    s = re.sub(r"\s+\'", "'", s)
    return s


# In[79]:


df = pd.read_json(f'../data/wiki-pages/wiki-062.jsonl', orient='records', lines=True)


# In[80]:


df['clean'] = df['text'].apply(lambda x: clean(x))
entities = df['clean'].apply(lambda x: re.findall(woa, x)).explode().dropna()
d = pd.DataFrame(list(entities))
d = d[0].apply(lambda x: re.sub(r'\(.*', "", x))
d = d.apply(lambda x: x.strip())
e = d[d.apply(lambda x: len(str(x)) <= 20)]


# In[81]:


e


# In[52]:


df['text'][0]


# In[31]:


clean("Man 's Lust for Gold")


# In[ ]:




