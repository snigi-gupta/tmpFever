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

LogHelper.setup()
logger = LogHelper.get_logger(__name__)  # pylint: disable=invalid-name


def tf_idf_sim(claim, lines, tf):
    test = []
    for line in lines:
        test.append({"claim": claim, "text": line})

    return tf.lookup(test).reshape(-1).tolist()


def append_to_file(sample, fp):
    with open(fp, 'a+') as f:
        j = json.dumps(sample)
        f.write(str(j) + '\n')

def get_line_from_doc(page, line_no, db):
    lines = db.get_doc_lines(page)
    #print("***************LINES:", lines)
    lines = [line.split("\t")[1] if len(line.split("\t")[1]) > 1 else "" for line in lines.split("\n")]
    #all_lines.extend(list(filter(lambda x: x != "", lines)))
    return lines[line_no]

if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    logger = LogHelper.get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_sentences', type=int, default=5, help='number of top ranking sentences to save for each claim', )
    parser.add_argument('--db', type=str, default='data/fever/fever.db', help='/path/to/saved/db.db', )
    parser.add_argument('--output', type=str, default='data/tmp/nei_preds_dev.jsonl', help='file to write predicted sentences to', )
    parser.add_argument('--docs', type=str, default='data/tmp/nei_doc_preds_dev.jsonl', help='file containing the predicted documents', )
    parser.add_argument('--dataset', type=str, default='data/fever-data/dev.jsonl', help='training dataset', )
    parser.add_argument('--train_ds', type=str, default='data/fever-data/train.jsonl', help='training dataset', )
    parser.add_argument('--dev_ds', type=str, default='data/fever-data/dev.jsonl', help='training dataset', )
    parser.add_argument("--cuda-device", type=int, required=False,  default=0, help='id of GPU to use (if any)')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.realpath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    ####### BASELINE CODE. COMMENT IF NOT NEEDED #######
    logger.info("Load DB")
    db = FeverDocDB(args.db)
    jlr = JSONLineReader()
    formatter = FEVERGoldFormatter(set(), FEVERLabelSchema())

    logger.info("Read datasets")
    train_ds = DataSet(file=args.train_ds, reader=jlr, formatter=formatter)
    dev_ds = DataSet(file=args.dev_ds, reader=jlr, formatter=formatter)

    train_ds.read()
    dev_ds.read()

    logger.info("Generate vocab for TF-IDF")
    tf = XTermFrequencyFeatureFunction(db)
    tf.inform(train_ds.data, dev_ds.data)
    ####################################################


    ##### SENTENCE RETREIVAL IMPLEMENTATION ############
    sen_retriever = TfidfSentenceRetriever()
    ####################################################

    with open(args.docs, 'r') as doc_preds, open(args.dataset, "r") as dset:
        for line in tqdm.tqdm(dset):
            #print("____________________________________________")
            sample = json.loads(line)
            claim = sample["claim"]
            evidence = sample["evidence"]
            #lines_field = sample["lines"]
            if sample["verifiable"] == "VERIFIABLE":
                for i in range(len(evidence)):
                    ev_set = evidence[i]
                    for j in range(len(ev_set)):
                        ev_sen = ev_set[j]
                        article = ev_sen[2]
                        line_no = ev_sen[3]
                        doc_line = get_line_from_doc(article, line_no, db)
                        #print(f"claim: {claim}, article: {article}, line_no: {line_no}")
                        #print(f"retreived: {doc_line}")
                        sample["evidence"][i][j][3] = doc_line
                        #print(f"new sample: {sample}")
                append_to_file(sample, args.output)
                continue

            
            #print(f"&&&&&&&&&&&&&&NEI claim: {claim}")
            lines_field = json.loads(doc_preds.readline())["lines"]
            scores = tf_idf_sim(claim, lines_field, tf)

            scores = list(zip(scores, lines_field))
            scores = list(filter(lambda score: len(score[1].strip()), scores))
            sentences_l = list(sorted(scores, reverse=True, key=lambda elem: elem[0]))

            sentences = [s[1] for s in sentences_l[:args.n_sentences]]
            #print("Sentences: ", sentences)
            sample["evidence"] = [[[None, None, "Article_id", sen] for sen in sentences]]
            #print(sample["evidence"])
            append_to_file(sample, args.output)

