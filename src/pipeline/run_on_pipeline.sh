#!/bin/bash
# run_on_pipeline: runs a claim on the pipeline that has already been initalized and running
# created on 2 May 2020

######CONSTANTS

PIPELINE_DIR="/home/kikuchio/git_repos/635-nlp-group-project/tmpFever/src/pipeline"
PREP_SCRIPT="${PIPELINE_DIR}/prep_pipeline_data.py"
PREP_DATA="${PIPELINE_DIR}/../data/tmp/pipeline_data.jsonl"

#GCP_INSTANCE_NAME="instance-1"
#GCP_INSTANCE_ZONE="us-east1-c"
GCP_INSTANCE_NAME="instance-2"
GCP_INSTANCE_ZONE="us-east1-d"
VM_INPUT_DIR="~/nlp/src/data/tmp"
VM_PREDS_FILE="nlp/src/data/tmp/predictions.jsonl"

#####FUNCTIONS

usage() {
	echo "controller: entry point that sets off the series of commands that run the pipeline remotely"
	echo -e "\nUsage: controller.sh [CLAIM]"
}

clean_up() {
	exit 0
}

send_to_vm() {
	gcloud compute scp --recurse $* "$GCP_INSTANCE_NAME":"$VM_INPUT_DIR" --zone "$GCP_INSTANCE_ZONE"
}

get_preds() {
  gcloud compute scp --recurse "$GCP_INSTANCE_NAME":~/"$VM_PREDS_FILE" . --zone "$GCP_INSTANCE_ZONE"
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

#sleep 2

get_preds

cat "predictions.jsonl"
