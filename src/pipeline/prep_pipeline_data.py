#!/usr/bin/env python

import argparse
import json
import os

##### MAIN


def to_jsonl(data, fp):
    with open(fp, "w+") as f:
        f.write(json.dumps(data) + '\n')


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--claim', type=str, help='the claim to run through the pipeline', )
    parser.add_argument('--output', type=str, default=f'{script_dir}/../data/tmp/pipeline_data.jsonl', help='the file to which the output should be written', )
    args = parser.parse_args()

    data = {}
    data["id"] = 73
    data["claim"] = args.claim

    to_jsonl(data, args.output)


    

