#!/usr/bin/env python3
from retrieval.tmp.document.drqa_doc_retriever import DrqaDocRetriever
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB

import argparse

from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
# from scripts.retrieval.sentence.process_tfidf import XTermFrequencyFeatureFunction

LogHelper.setup()
logger = LogHelper.get_logger(__name__)  # pylint: disable=invalid-name


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

    db = FeverDocDB(args.db)
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
