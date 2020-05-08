import pandas as pd
import spacy
import numpy as np

nlp = spacy.load('en_core_web_sm')
df = pd.read_json('../data/wiki-pages/wiki-001.jsonl',
                  orient='records', lines=True)

df['text'] = df['text'].replace('', np.nan)
df.dropna(subset=['text'], inplace=True)


def find_entities(s):
    doc = nlp(s)
    ents = doc.ents
    result = []
    for ent in ents:
        result.append([ent.label_, str(ent.text)])
    return result


entities = df.apply(lambda x: find_entities(x))
entities_values = entities.explode()
entities_index = entities.index

entities_df = pd.DataFrame(entities_values, columns=['label', 'value'])
entities_df['doc_id'] = entities_index

