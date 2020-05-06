from django.shortcuts import render
import subprocess
import os
import json
import time

fever_pipeline = "./run_on_pipeline.sh"
pred_file = "./predictions.jsonl"

def index(request):
    return render(request, "fever/index.html", {})


def process_claim(request):
    claim = request.POST['claim']
    print("claim is: ", claim)
    resp = verify(claim)
    resp["claim"] = claim
    return render(request, "fever/index.html", resp)


def verify(claim):
    proc = subprocess.run([fever_pipeline, claim])
    while not os.path.exists(pred_file):
        pass
    time.sleep(0.1)
    d = {}
    ev_pairs = []
    with open(pred_file, 'r') as f:
        line = json.loads(f.readline())
        d["label"] = line["predicted_label"]
        for idx, sen in enumerate(line["predicted_evidence"], 0):
            #d["sen" + str(idx+1)] = sen
            ev_pairs.append(EvidencePair(sen[0], sen[1]))
    d["evidence_pairs"] = ev_pairs
    return d


class EvidencePair:

    def __init__(self, article, evidence):
        self.sentence = evidence
        self.article = article

# Create your views here.
