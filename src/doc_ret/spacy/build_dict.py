import spacy
import os
import argparse
import json
import glob

nlp = spacy.load('en_core_web_sm')
parser = argparse.ArgumentParser()
parser.add_argument('--wiki_pages', type=str, default='data/wiki-pages/', help='/path/to/wiki-pages', )
parser.add_argument('--output', type=str, default='data/doc_ret/', help='/path/to/output', )

args = parser.parse_args()
out_dir = os.path.dirname(os.path.realpath(args.output))
os.makedirs(out_dir, exist_ok=True)

wiki_pages = glob.glob(f'{args.wiki_pages}*')

entities = {}
limit = 5
i = 0

for file in wiki_pages[25:30]:
    print(f'Reading {file}')
    if i > limit:
        break
    f = open(file, 'rb')
    j = 0
    for line in f:
        print(f'{file} {j}')
        h = json.loads(line)
        doc = nlp(h['text'])
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = {}
            if ent.text not in entities[ent.label_]:
                entities[ent.label_][ent.text] = set()
            entities[ent.label_][ent.text].add(h['id'])
        j += 1
    f.close()
    i += 1
    print(f'Done {file}')

f = open(f'{out_dir}output_6.json', 'w')
f.write(json.dumps(entities))
