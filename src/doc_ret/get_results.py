#!/usr/bin/env python
# coding: utf-8

# In[122]:


from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import spacy
import re
import numpy as np
from datetime import datetime
import jsonlines
import io
import random


# In[123]:


get_ipython().run_line_magic('load_ext', 'autotime')


# In[124]:


client = MongoClient()
mdb = client.fever2
entitycollection = mdb['entity_titles']
nouncollection = mdb['noun_titles']
titles = mdb['titles']


# In[77]:


nlp = spacy.load('ner_model3')


# In[162]:


df = pd.read_json('../data/fever-data/dev.jsonl', orient='records', lines=True)


# In[163]:


batch_size = 50


# In[168]:


df['claim'] = df['claim'].apply(lambda x: re.sub(r"\'s", "", x))


# In[169]:


df['claim'][13]


# In[170]:


batches = np.array_split(df['claim'], batch_size)


# In[171]:


match = ['PERSON', 'POI', 'WORK_OF_ART']

def process_ent_results(c, query_to_claim):
    result = {}
    for ent in c:
        claims = query_to_claim[ent['text']]
        doc = ent['doc_id']
        if ent['type'] in match:
            score = 5
        else:
            score = 1
        for claim in claims:
            if claim not in result:
                result[claim] = {}
            if doc not in result[claim]:
                result[claim][doc] = score
    return result


def process_noun_results(c, query_to_claim):
    result = []
    for nc in c:
        claims = query_to_claim[nc['text']]
        doc = nc['doc_id']
        for claim in claims:
            if claim not in result:
                result[claim] = {}
            if doc not in result[claim]:
                result[claim][doc] = 2
    return result
    
def process_title_results(c, query_to_claim, result):
    for res in c:
        claims = query_to_claim[res['cleaned']]
        doc = res['doc_id']
        for claim in claims:
            if claim not in result:
                result[claim] = {}
            if doc not in result[claim]:
                result[claim][doc] = 0
            result[claim][doc] += 100
    return result

def create_search_queries(batch):
    i = 0
    indices = batch.index
    result = {}
    noun_result = {}
    for doc in nlp.pipe(batch):
        for ent in doc.ents:
            if str(ent.text) not in result:
                result[str(ent.text)] = []
            result[str(ent.text)].append(indices[i])
        for nc in doc.noun_chunks:
            if str(nc.text) not in result:
                result[str(nc.text)] = []
            result[str(nc.text)].append(indices[i])
            
            if str(nc.text) not in noun_result:
                noun_result[str(nc.text)] = []
            noun_result[str(nc.text)].append(indices[i])
        i += 1
    for key, val in result.items():
        result[key] = list(set(val))
    for key, val in noun_result.items():
        noun_result[key] = list(set(val))
    print(len(result.keys()))
    return [result, noun_result]


def rank_docs(claim_to_docs, result, top = 5):
    for key, val in claim_to_docs.items():
        result[key] = [k for k, v in sorted(val.items(), key=lambda item: item[1], reverse=True)][:top]
    return result


# In[172]:


b = 0
pred_docs = {}
for batch in batches:
    s1 = datetime.now()
    query_to_claim, noun_to_claim = create_search_queries(batch)
    e1 = datetime.now()
    print(f'batch {b} - to_search = {len(query_to_claim.keys())} - {e1 - s1}')
    
    s2 = datetime.now()
    c = entitycollection.find({'text': {'$in': list(query_to_claim.keys())}})
    result = process_ent_results(c, query_to_claim)
    e2 = datetime.now()
    print(f'batch {b} - got_ents - {e2 - s2}')
    
    c = titles.find({'cleaned': {'$in': list(noun_to_claim.keys())}})
    result = process_title_results(c, query_to_claim, result)
    e3 = datetime.now()
    
    pred_docs = rank_docs(result, pred_docs, 10)
    print(f'batch {b} done - {e3 - s1}')
    b += 1


# In[173]:


result = []
index = df.index
null_vals = 0
fill = ["Inna_Trazhukova", "Hippodamia_-LRB-horse-RRB-", "Barbara_Ovstedal", "Solidarity_-LRB-album-RRB-", "The_Ark,_London"]
for i in index:
    h = {
        'id': int(df['id'][i]),
        'verifiable': df['verifiable'][i],
        'claim': df['claim'][i],
        'evidence': df['evidence'][i]
    }
    e = []
    if i in pred_docs:
        docs = pred_docs[i]
    else:
        null_vals += 1
        docs = fill
    h['pred_evidence'] = docs
    result.append(h)


# In[174]:


fp = io.BytesIO()
writer = jsonlines.Writer(fp)
writer.write_all(result)
writer.close()

f = open('../data/tmp/doc_ret_dev5.jsonl', 'wb+')
f.write(fp.getbuffer())
f.close()
fp.close()


# In[83]:


list(pred_docs.keys())[0]


# In[146]:


result[11111]


# In[142]:


pred_docs[105]


# In[101]:


batch = pd.Series(['The 14th Dalai Lama lives in Japan exclusively.'])
query_to_claim, n = create_search_queries(batch)


# In[103]:


query_to_claim


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from nltk.corpus import stopwords
stop = stopwords.words('english')


# In[ ]:


pat = re.compile(r'\b(?:{0})\b'.format('|'.join(stop)), re.IGNORECASE)


# In[ ]:


df['clean_claim'] = df['claim'].replace(to_replace=pat, value="", regex=True)


# In[ ]:


def create_entities_to_claims(batch):
    i = 0
    indices = batch.index
    result = {}
    for doc in nlp.pipe(batch):
        for ent in doc.ents:
            if str(ent.text) not in result:
                result[str(ent.text)] = []
            result[str(ent.text)].append(indices[i])
        i += 1
    return result

def create_nouns_to_claims(batch):
    i = 0
    indices = batch.index
    result = {}
    for doc in nlp.pipe(batch, disable=['ner']):
        for nc in doc.noun_chunks:
            if str(nc.text) not in result:
                result[str(nc.text)] = []
            result[str(nc.text)].append(indices[i])
        i += 1
    return result

match = ['PERSON', 'POI', 'WORK_OF_ART']

def create_ents_to_claims(c, ents_to_claim):
    claim_to_docs = {}
    for ent in c:
        claims = ents_to_claim[ent['text']]
        doc = ent['doc_id']
        if ent['type'] in match:
            score = 5
        else:
            score = 1
        for claim in claims:
            if claim not in claim_to_docs:
                claim_to_docs[claim] = {}
            if doc not in claim_to_docs[claim]:
                claim_to_docs[claim][doc] = 0
                claim_to_docs[claim][doc] += score
    return claim_to_docs

def create_ns_to_claims(c, ents_to_claim, claim_to_docs):
    for nc in c:
        claims = ents_to_claim[nc['text']]
        doc = nc['doc_id']
        for claim in claims:
            if claim not in claim_to_docs:
                claim_to_docs[claim] = {}
            if doc not in claim_to_docs[claim]:
                claim_to_docs[claim][doc] = 0
            claim_to_docs[claim][doc] += 7
    return claim_to_docs

def rank_docs(claim_to_docs, result, top = 5):
    for key, val in claim_to_docs.items():
        result[key] = [k for k, v in sorted(val.items(), key=lambda item: item[1], reverse=True)][:top]
    return result


# In[ ]:





# In[ ]:


pred_docs


# In[ ]:


result[1194]


# In[ ]:





# In[ ]:


pd.DataFrame(['Telemundo is a English-language television network.'])


# In[ ]:


batch = pd.Series(['Telemundo is a English-language television network.'])


# In[ ]:


ents_to_claim = create_entities_to_claims(batch)


# In[ ]:


ents_to_claim


# In[ ]:


nouns_to_claim = create_nouns_to_claims(batch)


# In[ ]:


nouns_to_claim


# In[ ]:


c = entitycollection.find({'text': {'$in': list(ents_to_claim.keys())}})


# In[ ]:


claim_to_docs = create_ents_to_claims(c, ents_to_claim)


# In[ ]:


claim_to_docs


# In[ ]:


r = []
for i in c:
    r.append(i)


# In[ ]:


r


# In[ ]:


ents_to_claim.keys()


# In[ ]:


c = mdb['nouns'].find({'text': {'$in': list(res.keys())}})


# In[ ]:


nouncollection


# In[ ]:


res = create_nouns_to_claims(batches[0])


# In[ ]:


len(res.keys())


# In[ ]:


c = nouncollection.find({'text': {'$in': list(res.keys())}})


# In[ ]:


res['baseball']


# In[ ]:


r[42]


# In[ ]:


res = create_search_queries(batches[0])


# In[ ]:


list(res.keys())[190:210]


# In[ ]:


res['Sancho Panza']


# In[ ]:


res['a vegan']


# In[ ]:




