#!/bin/bash
# Runs the entire pipeline on the provided data
# Created on 1 May 2020

######CONSTANTS

PIPELINE_DIR="/home/kikuchio/nlp/src"
#INPUT_PATH="${PIPELINE_DIR}/data/tmp/pipeline_data.jsonl"
INPUT_PATH="data/tmp/pipeline_data.jsonl"
DOC_PREDS_OUTPUT="data/tmp/doc_preds.jsonl"
SEN_PREDS_OUTPUT="data/tmp/sen_preds.jsonl"
LOG_FILE="pipeline.log"

#####FUNCTIONS

usage() {
  echo "interactive_pipeline: run the entire pipeline, consisting of document retrieval,
  sentence selection, and rte on the provided data"
  echo -e "\nUsage: interactive_pipeline.sh"
}

clean_up() {
  exit 0
}

######MAIN

trap clean_up SIGINT SIGTERM SIGHUP

cd "$PIPELINE_DIR"
rm "$DOC_PREDS_OUTPUT" "$SEN_PREDS_OUTPUT" "$LOG_FILE" &> /dev/null
pkill -f -9 "ret_pipeline"
pkill -f -9 "active_preprocess_fever"
pkill -f -9 "active_test_fever"
pkill -f -9 "active_rte"

case "$1" in
  "--help")
    usage 
      exit 0
        ;;
esac

/opt/conda/bin/python -u -m pipeline.ret_pipeline --dataset "$INPUT_PATH" &>> "$LOG_FILE" &

/opt/conda/bin/python -u -m rte.coetaur0.scripts.preprocessing.active_preprocess_fever &>> "$LOG_FILE" &

/opt/conda/bin/python -u -m rte.coetaur0.scripts.testing.active_test_fever &>> "$LOG_FILE" &

/opt/conda/bin/python -u -m pipeline.active_rte &>> "$LOG_FILE" &

