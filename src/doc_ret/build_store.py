#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import re
import spacy
from spacy.pipeline import EntityRuler
import time
import datetime
import uuid


# In[2]:


get_ipython().run_line_magic('load_ext', 'autotime')


# In[3]:


nlp = spacy.load('ner_model3')


# In[4]:


from pymongo import MongoClient
client = MongoClient()
mdb = client.fever2
entities = mdb['entities']
entity_titles = mdb['entity_titles']
nouns = mdb['nouns']
noun_titles = mdb['noun_titles']


# In[5]:


from nltk.corpus import stopwords
stop = stopwords.words('english')
from pymongo.errors import BulkWriteError

reqd = ['PERSON', 'FAC', 'ORG', 'GPE', 'LOC', 'POI', 'EVENT', 'WORK_OF_ART']

def find_enitites_and_nouns(nlp, df, key):
    result = {'entities': [], 'nouns': []}
    i = 0
    for doc in nlp.pipe(df[key]):
        
        for ent in doc.ents:
            if ent.label_ in reqd:
                result['entities'].append({
                    'type': ent.label_,
                    'text': str(ent.text),
                    'doc_id': df['id'][i],
                    '_id': uuid.uuid4().hex
                })
        for chunk in doc.noun_chunks:
            result['nouns'].append({
                'text': str(chunk.text),
                'doc_id': df['id'][i],
                '_id': uuid.uuid4().hex
            })
        i += 1
    return result

def clean(s):
    s = s.replace('-LRB-', '(')
    s = s.replace('-LSB-', '[')
    s = s.replace('-RRB-', ')')
    s = s.replace('-RSB-', ']')
    s = s.replace('_', ' ')
    s = s.replace('-COLON-', ':')
    s = re.sub(r"\s+\'", "'", s)
    return s

def write_to_db(table, rows):
    total = len(rows)
    batch_size = 1000
    i = 0
    while total > 0:
        y = i * batch_size
        to_insert = rows[y: y + batch_size]
        try:
            table.insert_many(rows)
            total -= batch_size
            i += 1
        except BulkWriteError as bwe:
            print(bwe.details)
            raise


# In[8]:


files = ['070', '082', '063']


# In[9]:


for f in files:
    s = datetime.datetime.now()
    df = pd.read_json(f'../data/wiki-pages/wiki-{f}.jsonl', orient='records', lines=True)
    df['clean_titles'] = df['id'].apply(lambda x: clean(x))
    df['clean_text'] = df['text'].apply(lambda x: clean(x))
    result = find_enitites_and_nouns(nlp, df, 'clean_text')
    nouns.insert_many(result['entities'], ordered=False)
    entities.insert_many(result['entities'], ordered=False)
    result = find_enitites_and_nouns(nlp, df, 'clean_titles')
    entity_titles.insert_many(result['entities'], ordered=False)
    noun_titles.insert_many(result['nouns'], ordered=False)
    print(f'done {f} - {datetime.datetime.now() - s}')


# In[ ]:




