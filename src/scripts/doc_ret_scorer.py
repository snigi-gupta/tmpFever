import argparse
import json
import pdb

#parser = argparse.ArgumentParser()
#parser.add_argument('true_values')
#parser.add_argument('pred_values')

#args = parser.parse_args()
true_hash = {}

f = open('/home/anirudh/ub/cse635/tmpFever/src/data/fever-data/dev.jsonl', 'r')
for line in f:
    h = json.loads(line)
    if h['verifiable'] == "NOT VERIFIABLE":
        continue
    res = []
    for e in h['evidence']:
        _, _, page_id, _ = e[0]
        if page_id:
            res.append(page_id)
    true_hash[h['id']] = res
f.close()

pred_hash = {}
f = open('/home/anirudh/ub/cse635/tmpFever/src/data/tmp/doc_ret_dev5.jsonl', 'r')

for line in f:
    h = json.loads(line)
    if h['verifiable'] == "NOT VERIFIABLE":
        continue
    res = []
    pred_hash[h['id']] = h['pred_evidence']
f.close()

tp = 0
fp = 0
tn = 0
fn = 0

for key, tvalue in true_hash.items():
    pvalue = pred_hash[key]
    tp += len(set(tvalue) & set(pvalue))
    fp += len(set(pvalue) - set(tvalue))
    fn += len(set(tvalue) - set(pvalue))

print(tp)
print(fp)
print(fn)

precision = (tp) / (tp + fp)
recall = (tp) / (tp + fn)
f1 = 2 * ((precision * recall) / (precision + recall))
print(f'precision = {precision}')
print(f'recall = {recall}')
print(f'f1 = {f1}')
