import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('true_values')
parser.add_argument('pred_values')

# "page_ids": ["Noticiero_Telemundo", "Noticias_Telemundo", "KTMO-LP", "KTMO-LP", "Noticias_Telemundo"], "indices": [0, 0, 3, 0, 8]

# {"id": 91198,
# "verifiable": "NOT VERIFIABLE",
# "label": "NOT ENOUGH INFO",
# "claim": "Colin Kaepernick became a starting quarterback during the 49ers 63rd season in the National Football League.",
# "evidence": [[[108548, null, null, null]]]}
# [[289914, 283015, "Soul_Food_-LRB-film-RRB-", 0]]

args = parser.parse_args()

true_hash = {}

f = open(args.true_values)
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
f = open(args.pred_values)
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

print(f'DR Accuracy = {docs_matched / true_docs}')
print(f'DR Macro Precision = {docs_precision / true_docs}')
print(f'DR Macro Recall = {docs_recall / true_docs}')

print(f'SS Accuracy = {sen_matched / true_sen}')
print(f'SS Macro Precision = {sen_precision / true_docs}')
print(f'SS Macro Recall = {sen_recall / true_docs}')