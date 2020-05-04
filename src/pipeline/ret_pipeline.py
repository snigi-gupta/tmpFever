from retrieval.tmp.document.drqa_doc_retriever import DrqaDocRetriever
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB
from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
from retrieval.process_tfidf import XTermFrequencyFeatureFunction
from retrieval.tmp.sentence.tfidf_sentence_retriever import TfidfSentenceRetriever
import argparse
import json
import tqdm
import os

LogHelper.setup()
logger = LogHelper.get_logger(__name__)  # pylint: disable=invalid-name


def append_to_file(claim, claim_id, p_lines, db, output):
    lines_field = [pl[0] for pl in p_lines]
    line_indices_field = [pl[2] for pl in p_lines]
    pages_field = [pl[1] for pl in p_lines]

    res = {"id": claim_id, "claim": claim, "lines": lines_field, "indices": line_indices_field, "page_ids": pages_field}
    with open(output, 'a+') as out:
        j = json.dumps(res)
        out.write(str(j) + '\n')

def tf_idf_sim(claim, lines, tf):
    test = []
    for line in lines:
        test.append({"claim": claim, "text": line})

    return tf.lookup(test).reshape(-1).tolist()


def append_sens_to_file(claim, claim_id, sentences, page_ids, line_indices, fp):
    res = {"id": claim_id, "claim": claim, "evidence": sentences, "page_ids": page_ids, "indices": line_indices}
    with open(fp, 'a+') as f:
        j = json.dumps(res)
        f.write(str(j) + '\n')

if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    logger = LogHelper.get_logger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='data/fever/fever.db', help='/path/to/saved/db.db', )
    parser.add_argument('--n_docs', type=int, default=5, help='number of top ranking documents to save for claim', )
    parser.add_argument('--dataset', type=str, default='data/fever-data/test.jsonl', help='the train/dev/test dataset', )
    parser.add_argument('--output', type=str, default='data/tmp/doc_preds.jsonl', help='the file to which the output should be written', )
    parser.add_argument("--model",type=str, required=False,
                        default="data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz", help="model")
    parser.add_argument("--cuda-device", type=int, required=False,  default=0, help='id of GPU to use (if any)')

    # Sen
    parser.add_argument('--n_sentences', type=int, default=5, help='number of top ranking sentences to save for each claim', )
    parser.add_argument('--docs', type=str, default='data/tmp/doc_preds.jsonl', help='file containing the predicted documents', )
    parser.add_argument('--sen_output', type=str, default='data/tmp/sen_preds.jsonl', help='file to write predicted sentences to', )
    parser.add_argument('--train_ds', type=str, default='data/fever-data/train.jsonl', help='training dataset', )
    parser.add_argument('--dev_ds', type=str, default='data/fever-data/dev.jsonl', help='development dataset', )

    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.realpath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    ##### DOCUMENT RETREIVAL IMPLEMENTATION #####
    doc_retriever = DrqaDocRetriever(args.model)
    #############################################

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

    ##### SENTENCE RETREIVAL IMPLEMENTATION ############
    sen_retriever = TfidfSentenceRetriever()
    ####################################################

    while True:
        # wait for input
        while not os.path.exists(args.dataset):
            pass

        with open(args.dataset, 'r') as dset:
            for line in tqdm.tqdm(dset):
                sample = json.loads(line)
                claim = sample["claim"]
                claim_id = sample["id"]

                pages, _ = doc_retriever.closest_docs(claim, args.n_docs)
                p_lines = []
                for page in pages:
                    lines = db.get_doc_lines(page)
                    lines = [line.split("\t")[1] if len(line.split("\t")[1]) > 1 else "" for line in lines.split("\n")]
                    p_lines.extend(zip(lines, [page] * len(lines), range(len(lines))))
                append_to_file(claim, claim_id, p_lines, db, args.output)

            out_dir = os.path.dirname(os.path.realpath(args.output))
            os.makedirs(out_dir, exist_ok=True)

        with open(args.docs, 'r') as doc_preds:
            for line in tqdm.tqdm(doc_preds):
                sample = json.loads(line)
                claim = sample["claim"]
                claim_id = sample["id"]
                lines_field = sample["lines"]
                line_indices_field = sample["indices"]
                pages_field = sample["page_ids"]

                scores = tf_idf_sim(claim, lines_field, tf)

                scores = list(zip(scores, pages_field, line_indices_field, lines_field))
                scores = list(filter(lambda score: len(score[3].strip()), scores))
                sentences_l = list(sorted(scores, reverse=True, key=lambda elem: elem[0]))

                sentences = [s[3] for s in sentences_l[:args.n_sentences]]
                page_ids = [s[1] for s in sentences_l[:args.n_sentences]]
                line_indices = [s[2] for s in sentences_l[:args.n_sentences]]
                evidence = " ".join(sentences)
                print("Sentences: ", sentences)
                append_sens_to_file(claim, claim_id, sentences, page_ids, line_indices, args.sen_output)
        os.remove(args.dataset)

