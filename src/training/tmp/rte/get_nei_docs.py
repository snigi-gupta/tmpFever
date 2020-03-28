from retrieval.tmp.document.drqa_doc_retriever import DrqaDocRetriever
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB
import argparse
import json
import tqdm
import os

LogHelper.setup()
logger = LogHelper.get_logger(__name__)  # pylint: disable=invalid-name


def append_to_file(sample, output):
    with open(output, 'a+') as out:
        j = json.dumps(sample)
        out.write(str(j) + '\n')


if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    logger = LogHelper.get_logger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='data/fever/fever.db', help='/path/to/saved/db.db', )
    parser.add_argument('--n_docs', type=int, default=5, help='number of top ranking documents to save for claim', )
    parser.add_argument('--dataset', type=str, default='data/fever-data/dev.jsonl', help='the train/dev/test dataset', )
    parser.add_argument('--output', type=str, default='data/tmp/nei_doc_preds_dev.jsonl', help='the file to which the output should be written', )
    parser.add_argument("--model",type=str, required=False,
                        default="data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz", help="model")
    parser.add_argument("--cuda-device", type=int, required=False,  default=0, help='id of GPU to use (if any)')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.realpath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    ##### DOCUMENT RETREIVAL IMPLEMENTATION #####
    doc_retriever = DrqaDocRetriever(args.model)
    #############################################

    logger.info("Load DB")
    db = FeverDocDB(args.db)

    with open(args.dataset, 'r') as dset:
        for line in tqdm.tqdm(dset):
            sample = json.loads(line)
            claim = sample["claim"]
            claim_id = sample["id"]
            if sample["verifiable"] == "VERIFIABLE":
                continue

            pages, _ = doc_retriever.closest_docs(claim, args.n_docs)
            all_lines = []
            for page in pages:
                lines = db.get_doc_lines(page)
                lines = [line.split("\t")[1] if len(line.split("\t")[1]) > 1 else "" for line in lines.split("\n")]
                all_lines.extend(list(filter(lambda x: x != "", lines)))
            all_lines = {"claim": claim, "lines": all_lines}
            append_to_file(all_lines, args.output)

