from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from drqa import retriever
from retrieval.tmp.document.drqa_doc_retriever import DrqaDocRetriever
from allennlp.data import Tokenizer, TokenIndexer
from allennlp.models import Model, load_archive
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB
from rte.parikh.reader import FEVERReader

import argparse
import numpy as np


from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
# from scripts.retrieval.sentence.process_tfidf import XTermFrequencyFeatureFunction
from retrieval.process_tfidf import XTermFrequencyFeatureFunction


from rte.tmp.nei_rte_model import NeiRteModel

LogHelper.setup()
logger = LogHelper.get_logger(__name__)  # pylint: disable=invalid-name


def tf_idf_sim(claim, lines):
    test = []
    for line in lines:
        test.append({"claim": claim, "text": line})

    return tf.lookup(test).reshape(-1).tolist()

def eval_model(db: FeverDocDB, args) -> Model:
    # archive = load_archive(args.archive_file, cuda_device=args.cuda_device, overrides=args.overrides)

    # config = archive.config
    # ds_params = config["dataset_reader"]
    #
    # model = archive.model
    # model.eval()


    # reader = FEVERReader(db,
    #                              sentence_level=ds_params.pop("sentence_level",False),
    #                              wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})),
    #                              claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})),
    #                              token_indexers=TokenIndexer.dict_from_params(ds_params.pop('token_indexers', {})))

    model = NeiRteModel()
    reader = FEVERReader(db, sentence_level=False)


    while True:
        ############### CLAIM
        claim = input("enter claim (or q to quit) >>")
        if claim.lower() == "q":
            break

        ############### DOCUMENT RETRIEVAL
        # ranker = retriever.get_class('tfidf')(tfidf_path=args.model)
        # # ranker = retriever.get_class('tfidf')()
        #
        p_lines = []
        # pages,_ = ranker.closest_docs(claim,5)
        doc_retriever = DrqaDocRetriever(args.model)
        pages, _ = doc_retriever.closest_docs(claim, 5)
        print("Fetched Nearest 5 docs")

        for page in pages:
            lines = db.get_doc_lines(page)
            lines = [line.split("\t")[1] if len(line.split("\t")[1]) > 1 else "" for line in lines.split("\n")]

            p_lines.extend(zip(lines, [page] * len(lines), range(len(lines))))

        lines_field = [pl[0] for pl in p_lines]
        line_indices_field = [pl[2] for pl in p_lines]
        pages_field = [pl[1] for pl in p_lines]

        ############### SENTENCE RETRIEVAL

        # this line would be replaced by a call to the new implementation
        scores = tf_idf_sim(claim, lines_field)

        scores = list(zip(scores, pages_field, line_indices_field, lines_field))
        scores = list(filter(lambda score: len(score[3].strip()), scores))
        sentences_l = list(sorted(scores, reverse=True, key=lambda elem: elem[0]))

        sentences = [s[3] for s in sentences_l[:5]]
        evidence = " ".join(sentences)
        print("Sentences: ", sentences)

        ############### RTE
        print("Best pages: {0}".format(repr(pages)))

        print("Evidence:")
        for idx,sentence in enumerate(sentences_l[:5]):
            print("{0}\t{1}\t\t{2}\t{3}".format(idx+1, sentence[0], sentence[1],sentence[3]) )

        item = reader.text_to_instance(evidence, claim)

        print(f"item: {item}")
        prediction = model.forward(item)
        # prediction = model.forward_on_instance(item, args.cuda_device)
        # cls = model.vocab._index_to_token["labels"][np.argmax(prediction["label_probs"])]
        # print("PREDICTED: {0}".format(cls))
        print("PREDICTED: {0}".format(prediction))
        print("___________________________________")


if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    logger = LogHelper.get_logger(__name__)


    parser = argparse.ArgumentParser()

    parser.add_argument('--db', type=str, default='data/fever/fever.db', help='/path/to/saved/db.db', )
    # parser.add_argument('archive_file', type=str, default='', help='/path/to/saved/db.db')
    parser.add_argument("--model",type=str, required=False,
                        default="data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz", help="model")
    parser.add_argument("--cuda-device", type=int, required=False,  default=0, help='id of GPU to use (if any)')
    parser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a HOCON structure used to override the experiment configuration')



    args = parser.parse_args()

    doc_retriever = DrqaDocRetriever(args.model)
    while True:
        ############### CLAIM
        claim = input("enter claim (or q to quit) >>")
        if claim.lower() == "q":
            break
        ############### DOCUMENT RETRIEVAL
        # ranker = retriever.get_class('tfidf')(tfidf_path=args.model)
        # # ranker = retriever.get_class('tfidf')()
        #
        p_lines = []
        # pages,_ = ranker.closest_docs(claim,5)
        pages, _ = doc_retriever.closest_docs(claim, 5)
        print("Fetched Nearest 5 docs")
        print(pages)



    logger.info("Load DB")
    db = FeverDocDB(args.db)

    jlr = JSONLineReader()
    formatter = FEVERGoldFormatter(set(), FEVERLabelSchema())

    logger.info("Read datasets")
    # train_ds = DataSet(file="data/fever/train.ns.pages.p{0}.jsonl".format(1), reader=jlr, formatter=formatter)
    # dev_ds = DataSet(file="data/fever/dev.ns.pages.p{0}.jsonl".format(1), reader=jlr, formatter=formatter)
    #train_ds = DataSet(file="data/fever-data/train.jsonl", reader=jlr, formatter=formatter)
    #dev_ds = DataSet(file="data/fever-data/dev.jsonl", reader=jlr, formatter=formatter)

    #train_ds.read()
    #dev_ds.read()

    logger.info("Generate vocab for TF-IDF")
    tf = XTermFrequencyFeatureFunction(db)
    # tf.inform(train_ds.data, dev_ds.data)

    logger.info("Eval")
    eval_model(db,args)
