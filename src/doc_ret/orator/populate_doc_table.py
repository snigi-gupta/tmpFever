import pandas as pd
import argparse
import glob
from doc_ret.db import db
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--wiki_pages', type=str, default='data/wiki-pages/',
                    help='/path/to/wiki-pages', )
# parser.add_argument('--chunks', type=int, default='11', help='Number of files to work at once', )
# parser.add_argument('--chunk_id', type=int, help='Batch number to work on', )

args = parser.parse_args()
wiki_pages = glob.glob(f'{args.wiki_pages}*')

df = pd.DataFrame()

for f in wiki_pages:
    d = pd.read_json(f, orient='records', lines=True)
    del d['lines']
    del d['text']
    df = df.append(d, ignore_index=True)


rows = df.to_dict(orient='records')
total = len(rows)
batch_size = 1000
i = 0
while total > 0:
    y = i * batch_size
    to_insert = rows[y: y + batch_size]
    db.table('docs').insert(to_insert)
    print(i)
    total -= batch_size
    i += 1
