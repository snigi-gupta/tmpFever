#!/bin/bash
# controller: entry point that sets off the series of commands that run the pipeline remotely
# created on 1 May 2020

######CONSTANTS

PIPELINE_DIR="/home/kikuchio/git_repos/635-nlp-group-project/tmpFever/src/pipeline"
PREP_SCRIPT="${PIPELINE_DIR}/prep_pipeline_data.py"
PREP_DATA="${PIPELINE_DIR}/../data/tmp/pipeline_data.jsonl"

VM_INPUT_DIR="~/nlp/src/data/tmp"
VM_PIPELINE="~/nlp/src/pipeline/interactive_pipeline.sh"

#####FUNCTIONS

usage() {
	echo "controller: entry point that sets off the series of commands that run the pipeline remotely"
	echo -e "\nUsage: controller.sh [CLAIM]"
}

clean_up() {
	exit 0
}

send_to_vm() {
	gcloud compute scp --recurse $* instance-1:"$VM_INPUT_DIR" --zone us-east1-c;
}

start_pipeline() {
  gcloud compute ssh instance-1 --zone us-east1-c --command "export PYTHONPATH=/home/kikuchio/nlp/src/ && $VM_PIPELINE"
}

######MAIN

trap clean_up SIGINT SIGTERM SIGHUP

case "$1" in
	"--help")
					usage 
					exit 0
					;;
esac

if [ -z "$1" ]; then
	echo "please provide the claim"
	exit 1
fi

# prep the data for the first stage of the pipeline
python -u "$PREP_SCRIPT" --claim "$1" --output "$PREP_DATA"

send_to_vm "$PREP_DATA" 

start_pipeline
