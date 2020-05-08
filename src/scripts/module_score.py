import os
import argparse
import json

true_hash = {}

f = open('/home/anirudh/ub/cse635/tmpFever/src/data/fever-data/dev.jsonl', 'r')
for line in f:
	h = json.loads(line)
	if h['verifiable'] == "NOT VERIFIABLE":
		continue
	res = {}
	for e in h['evidence']:
		_, _, page_id, sentence_index = e[0]
		if page_id not in res:
			res[page_id] = []
		res[page_id].append(sentence_index)
	true_hash[h['id']] = res
f.close()

pred_hash = {}
f = open('/home/anirudh/ub/cse635/tmpFever/src/data/tmp/doc_preds.jsonl', 'r')
for line in f:
	h = json.loads(line)
	if h['id'] not in true_hash:
		continue
	page_sentence_id = {}
	page_ids = h['page_ids']
	sentence_indices = h['indices']
	for i in range(0, len(page_ids)):
		if page_ids[i] not in page_sentence_id:
			page_sentence_id[page_ids[i]] = []
		page_sentence_id[page_ids[i]].append(sentence_indices[i])
	pred_hash[h['id']] = page_sentence_id
f.close()

docs_matched = 0
true_docs = 0

docs_precision = 0
docs_recall = 0

sen_matched = 0
true_sen = 0

sen_precision = 0
sen_recall = 0

'''
tp = 0
fp = 0
tn = 0
fn = 0

for key, tvalue in true_hash.items():
    pvalue = pred_hash[key]
    tp += len(set(tvalue.keys()) & set(pvalue.keys()))
    fp += len(set(pvalue.keys()) - set(tvalue.keys()))
    fn += len(set(tvalue.keys()) - set(pvalue.keys()))

precision = (tp) / (tp + fp)
recall = (tp) / (tp + fn)
f1 = 2 * ((precision * recall) / (precision + recall))
print(f'precision = {precision}')
print(f'recall = {recall}')
print(f'f1 = {f1}')
'''

for key, tvalue in true_hash.items():
	pvalue = pred_hash[key]
	docs_matched += len(set(tvalue.keys()) & set(pvalue.keys()))
	true_docs += len(set(tvalue.keys()))

	docs_precision += (len(set(tvalue.keys()) & set(pvalue.keys())) / (len(set(tvalue.keys()).union(set(pvalue.keys())))))
	if set(pvalue.keys()).issubset(set(tvalue.keys())):
		docs_recall += 1

	for page, tindices in tvalue.items():
		true_sen += len(set(tindices))
		if page in pvalue:
			pindices = pvalue[page]
			sen_matched += len(set(tindices) & set(pindices))
			sen_precision += (len(set(tindices) & set(pindices)) / (len(set(tindices).union(set(pindices)))))
			if set(pindices).issubset(set(tindices)):
				sen_recall += 1

print(f'|V| = {true_docs}')
print(f'{docs_recall}')
print(f'DR Accuracy = {docs_matched / true_docs}')
print(f'DR Macro Precision = {docs_precision / true_docs}')
print(f'DR Macro Recall = {docs_recall / true_docs}')

print(f'SS Accuracy = {sen_matched / true_sen}')
print(f'SS Macro Precision = {sen_precision / true_docs}')
print(f'SS Macro Recall = {sen_recall / true_docs}')
