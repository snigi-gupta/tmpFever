#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autotime')


# In[26]:


import pandas as pd
import spacy
import numpy as np
import json
import re
import uuid
from pymongo import MongoClient


# In[27]:


client = MongoClient()
mdb = client.fever2


# In[14]:


woa_list = [
    'album', 'game', 'film', 'song',
    'book', 'series', 'TV', 'novel',
    'EP', 'comic', 'serial', 'tale',
    'play', 'movie', 'anime', 'comic',
    'journal', 'show'
]
poi_list = ['album', 'song', 'singer']
sports_list = [
    'soccer', 'football', 'cricket',
    'basketball', 'hockey', 'baseball',
    'playoff', 'team', 'championship',
    'finals', 'final', 'play-off',
    'play-offs', 'playoffs'
]


# In[15]:


woa = re.compile(r'\b(?:{0})\b'.format('|'.join(woa_list)))
poi = re.compile(r'([A-Z]\w+.*) (\b(?:{0})\b)'.format('|'.join(poi_list)))
sports = re.compile(r'\b(?:{0})\b'.format('|'.join(sports_list)))
groupr = r'(.*) \((.*)\)'


# In[16]:


def find_entities(s):
    groups = re.findall(groupr, s)
    result = []
    if len(groups) == 1:
        text, disamb = groups[0]
        if disamb and not disamb[0].isdigit() and len(re.findall(woa, disamb)) > 0 and len(text) > 3:
            result.append({
                'label': 'WORK_OF_ART',
                'pattern': text
            })
            disambgroups = re.findall(poi, disamb)
            if len(disambgroups) == 1:
                t, _ = disambgroups[0]
                if len(t) > 3:
                    result.append({
                        'label': 'POI',
                        'pattern': t
                    })
    return result


# In[28]:


def clean(s):
    s = s.replace('-LRB-', '(')
    s = s.replace('-LSB-', '[')
    s = s.replace('-RRB-', ')')
    s = s.replace('-RSB-', ']')
    s = s.replace('_', ' ')
    s = s.replace('-COLON-', ':')
    s = re.sub(r"\s+\'", "'", s)
    s = re.sub(r'\(.*', "", s)
    s = s.strip()
    return s

def read_and_get_titles(fname):
    df = pd.read_json(f'../data/wiki-pages/wiki-{fname}.jsonl', orient='records', lines=True)
    df['_id'] = df['id'].apply(lambda x: uuid.uuid4().hex)
    df['doc_id'] = df['id']
    df['cleaned'] = df['id'].apply(lambda x: clean(x))
    return df
    
def read_and_build_custom_entities(fname):
    df = pd.read_json(f'../data/wiki-pages/wiki-{fname}.jsonl', orient='records', lines=True)
    df['cleaned'] = df['id'].apply(lambda x: clean(x))
    patterns = list(df['cleaned'].apply(lambda x: find_entities(x)).explode().dropna())
    df['clean'] = df['text'].apply(lambda x: clean(x))
    entities = df['clean'].apply(lambda x: re.findall(woa, x)).explode().dropna()
    entities = pd.DataFrame(list(entities))
    entities = entities[0].apply(lambda x: re.sub(r'\(.*', "", x))
    entities = entities.apply(lambda x: x.strip())
    entities = entities[entities.apply(lambda x: len(str(x)) <= 20)]
    for x in entities:
        patterns.append({'label': 'WORK_OF_ART', 'pattern': x})
    print(f'done {fname}')
    return patterns


# In[30]:


patterns = []

files = [
    '029', '073', '098', '001',
    '004', '079', '039', '050',
    '095', '038', '065', '012',
    '024', '092', '022', '075',
    '088', '021', '074', '009',
    '016', '086', '064', '061',
    '025', '057', '037', '054',
    '030', '036', '069', '026',
    '007', '035', '048', '047',
    '042', '109', '044', '019',
    '067', '056', '103', '094',
    '104', '062', '068', '032',
    '023', '066', '084', '013',
    '072', '010', '041', '045',
    '014', '052', '018', '033',
    '105', '077', '053', '070',
    '082', '063', '106', '100',
    '102', '020', '093', '015',
    '090', '080', '058', '017',
    '028', '078', '083', '089',
    '085', '091', '027', '099',
    '002', '087', '081', '003',
    '101', '005', '051', '059',
    '071', '097', '046', '011',
    '107', '008', '055', '076',
    '049', '006', '108', '060',
    '096', '043', '034', '031', '040'
]


# In[31]:


for f in files:
    rows = read_and_get_titles(f)[['doc_id', 'cleaned', '_id']]
    mdb['titles'].insert_many(rows.to_dict('records'))
    print(f'done {f}')


# In[8]:


for fname in files:
    patterns.extend(read_and_build_custom_entities(fname))


# In[9]:


len(patterns)


# In[12]:


uniq_patterns = {}
for pattern in patterns:
    if pattern['pattern'] not in uniq_patterns:
        uniq_patterns[pattern['pattern']] = pattern


# In[13]:


uniq_patterns = list(uniq_patterns.values())
len(uniq_patterns)


# In[14]:


import spacy
from spacy.pipeline import EntityRuler


# In[15]:


nlp = spacy.load("en_core_web_lg")


# In[16]:


doc = nlp("True Story (The B.G.'z album)")
doc.ents


# In[17]:


doc = nlp('Mad Max : Fury Road')
doc.ents


# In[18]:


ruler = EntityRuler(nlp)


# In[19]:


batches = np.array_split(uniq_patterns, 10)


# In[21]:


for batch in batches:
    ruler.add_patterns(batch)


# In[22]:


nlp.add_pipe(ruler, before='ner')


# In[23]:


doc = nlp("True Story (The B.G.'z album)")
doc.ents


# In[24]:


doc = nlp("I")
doc.ents


# In[25]:


doc = nlp('Mad Max : Fury Road')
doc.ents


# In[34]:


nlp.to_disk('ner_model3')


# In[49]:


doc = nlp("The Challenge: Rivals III is a season of American Ninja Warrior.")
doc.ents


# In[50]:


for w in doc.noun_chunks:
    print(w.text, w.root.dep_)


# In[45]:


doc = nlp("Heli Simpson")
doc.ents[0].label_


# In[31]:


batches[0][1000]


# In[44]:


doc[0].tag


# In[ ]:




