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


def tf_idf_sim(claim, lines, tf):
    test = []
    for line in lines:
        test.append({"claim": claim, "text": line})

    return tf.lookup(test).reshape(-1).tolist()


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
    parser.add_argument('--n_sentences', type=int, default=5, help='number of top ranking sentences to save for each claim', )
    parser.add_argument('--model', type=str, required=False, help='path to pretrained model', )
    parser.add_argument('--db', type=str, default='data/fever/fever.db', help='/path/to/saved/db.db', )
    parser.add_argument('--sens', type=str, default='data/tmp/sen_preds.jsonl', help='file containing the predicted documents', )
    parser.add_argument('--rte_preds', type=str, default='rte/coetaur0/data/test_preds.txt', help='file containing the predicted labels', )
    parser.add_argument('--output', type=str, default='data/tmp/predictions.jsonl', help='file to write predicted sentences to', )
    parser.add_argument('--train_ds', type=str, default='data/fever-data/train.jsonl', help='training dataset', )
    parser.add_argument('--dev_ds', type=str, default='data/fever-data/dev.jsonl', help='development dataset', )
    parser.add_argument("--cuda-device", type=int, required=False,  default=0, help='id of GPU to use (if any)')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.realpath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(args.output):
        os.remove(args.output)

    ####### BASELINE CODE. COMMENT IF NOT NEEDED #######
    #logger.info("Load DB")
    #db = FeverDocDB(args.db)
    #jlr = JSONLineReader()
    #formatter = FEVERGoldFormatter(set(), FEVERLabelSchema())

    #logger.info("Read datasets")
    #train_ds = DataSet(file=args.train_ds, reader=jlr, formatter=formatter)
    #dev_ds = DataSet(file=args.dev_ds, reader=jlr, formatter=formatter)

    #train_ds.read()
    #dev_ds.read()

    #logger.info("Generate vocab for TF-IDF")
    #tf = XTermFrequencyFeatureFunction(db)
    #tf.inform(train_ds.data, dev_ds.data)
    ####################################################


    ##### RTE IMPLEMENTATION ############
    #rte_impl = NeiRteModel
    #rte_impl = RnnRteModel
    #if args.model:
    #    logger.info("loading saved model...")
    #    rte_model = rte_impl.from_state(args.model, args.cuda_device)
    #else:
    #    rte_model = rte_impl(args.cuda_device)
    ####################################################

    while True:
        while not os.path.exists(args.rte_preds):
            pass
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


