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


# In[3]:


nlp = spacy.load('ner_model3')


# In[4]:


from nltk.corpus import stopwords
stop = stopwords.words('english')

def find_enitites_and_nouns(nlp, df):
    result = {'entities': [], 'nouns': []}
    i = 0
    for doc in nlp.pipe(df['text']):
        for ent in doc.ents:
            result['entities'].append({
                'type': ent.label_,
                'text': str(ent.text),
                'doc_id': df['id'][i]
            })
        for chunk in doc.noun_chunks:
            result['nouns'].append({'text': str(chunk.text), 'doc_id': df['id'][i]})
        i += 1
    return result

def find_enitites_in_title(nlp, df):
    result = []
    i = 0
    for doc in nlp.pipe(df['cleaned']):
        for ent in doc.ents:
            result.append({
                'type': ent.label_,
                'text': str(ent.text),
                'doc_id': df['id'][i]
            })
        i += 1
    return result

def clean(s):
    s = s.replace('-LRB-', '(')
    s = s.replace('-LSB-', '[')
    s = s.replace('-RRB-', ')')
    s = s.replace('-RSB-', ']')
    s = s.replace('_', ' ')
    return s

def write_to_db(table, rows):
    total = len(rows)
    batch_size = 500
    i = 0
    while total > 0:
        y = i * batch_size
        to_insert = rows[y: y + batch_size]
        db.table(table).insert(to_insert)
        total -= batch_size
        i += 1


# In[5]:


files = ['064', '061', '025', '057', '037', '054', '030', '036', '069',
       '026', '007', '035', '048', '047', '042', '109', '044', '019',
       '067', '056', '103', '094']


# In[8]:


pat = re.compile(r'\b(?:{0})\b'.format('|'.join(stop)), re.IGNORECASE)
import datetime
for fname in files:
    start = datetime.datetime.now()
    df = pd.read_json(f'../data/wiki-pages/wiki-{fname}.jsonl', orient='records', lines=True)
    df['cleaned'] = df['id'].apply(lambda x: clean(x))
    df['text'] = df['text'].replace(to_replace=pat, value="", regex=True)
    rows = pd.DataFrame()
    rows['page_id'] = df['id']
    rows['text'] = df['text']
    rows['lines'] = df['lines']
    rows = rows.to_dict(orient='records')
    write_to_db('docs', rows)
    del rows
    
    result = find_enitites_and_nouns(nlp, df)
    write_to_db('entities', result['entities'])
    write_to_db('nouns', result['nouns'])
    del result
    
    result = find_enitites_in_title(nlp, df)
    write_to_db('titles', result)
    del result
    
    print(f'done {fname} - {datetime.datetime.now() - start}')


# In[ ]:




