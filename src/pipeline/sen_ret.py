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


def append_to_file(claim, sentences, fp):
    res = {"claim": claim, "sentences": sentences}
    with open(fp, 'a+') as f:
        j = json.dumps(res)
        f.write(str(j) + '\n')

if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    logger = LogHelper.get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_sentences', type=int, default=5, help='number of top ranking sentences to save for each claim', )
    parser.add_argument('--db', type=str, default='data/fever/fever.db', help='/path/to/saved/db.db', )
    parser.add_argument('--docs', type=str, default='data/tmp/doc_preds.jsonl', help='file containing the predicted documents', )
    parser.add_argument('--output', type=str, default='data/tmp/sen_preds.jsonl', help='file to write predicted sentences to', )
    parser.add_argument('--train_ds', type=str, default='data/fever-data/train.jsonl', help='training dataset', )
    parser.add_argument('--dev_ds', type=str, default='data/fever-data/dev.jsonl', help='development dataset', )
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

    with open(args.docs, 'r') as doc_preds:
        for line in tqdm.tqdm(doc_preds):
            sample = json.loads(line)
            claim = sample["claim"]
            lines_field = sample["lines"]
            line_indices_field = sample["indices"]
            pages_field = sample["page_ids"]

            ###### COMMENT THIS LINE WHEN IMPLEMENTATION IS READY ######
            scores = tf_idf_sim(claim, lines_field, tf)

            ###### UNCOMMENT THIS LINE WHEN IMPLEMENTATION IS READY ######
            #scores = sen_retriever.score_sentences(claim, lines_field)

            scores = list(zip(scores, pages_field, line_indices_field, lines_field))
            scores = list(filter(lambda score: len(score[3].strip()), scores))
            sentences_l = list(sorted(scores, reverse=True, key=lambda elem: elem[0]))

            sentences = [s[3] for s in sentences_l[:args.n_sentences]]
            evidence = " ".join(sentences)
            print("Sentences: ", sentences)
            append_to_file(claim, sentences, args.output)



