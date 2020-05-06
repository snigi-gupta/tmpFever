from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
from retrieval.process_tfidf import XTermFrequencyFeatureFunction
from retrieval.tmp.sentence.tfidf_sentence_retriever import TfidfSentenceRetriever
import argparse
import tqdm
import json
import os
from rte.tmp.nei_rte_model import NeiRteModel
from rte.tmp.rnn_rte_model import RnnRteModel
import time

LogHelper.setup()
logger = LogHelper.get_logger(__name__)  # pylint: disable=invalid-name


def append_to_file(claim_id, prediction, evidence, fp):
    res = {"id": claim_id, "predicted_label": prediction, "predicted_evidence": evidence}
    with open(fp, 'w+') as f:
        j = json.dumps(res)
        f.write(str(j) + '\n')

if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    logger = LogHelper.get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--sens', type=str, default='data/tmp/sen_preds.jsonl', help='file containing the predicted documents', )
    parser.add_argument('--rte_preds', type=str, default='rte/coetaur0/data/test_preds.txt', help='file containing the predicted labels', )
    parser.add_argument('--output', type=str, default='data/tmp/predictions.jsonl', help='file to write predicted sentences to', )
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.realpath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(args.output):
        os.remove(args.output)

    # keep waiting for input
    while True:
        # keep waiting for pipeline to finish processing the input
        while not os.path.exists(args.rte_preds):
            pass
        # waiting until the contents of the file are complete
        time.sleep(0.2)
        logger.info("processing predicted sentences")
        with open(args.sens, 'r') as sen_preds, open(args.rte_preds, 'r') as rte_preds:
            for line in tqdm.tqdm(sen_preds):
                sample = json.loads(line)
                claim = sample["claim"]
                claim_id = sample["id"]
                sentences = sample["evidence"]
                line_indices_field = sample["indices"]
                pages_field = sample["page_ids"]

                pred = rte_preds.readline().strip()
                if pred == "2":
                    pred = "NOT ENOUGH INFO"
                elif pred == "1":
                    pred = "REFUTES"
                elif pred == "0":
                    pred = "SUPPORTS"
                print("rte pred: ", pred)

                evidence = [[page_id, ev] for page_id, ev in zip(pages_field, sentences)]

                append_to_file(claim_id, pred, evidence, args.output)
        os.remove(args.sens)
        os.remove(args.rte_preds)


