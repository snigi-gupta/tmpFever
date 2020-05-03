#!/bin/bash
# start_modules: entry point that sets off the series of commands that start the pipeline modules remotely
# created on 1 May 2020

######CONSTANTS

#GCP_INSTANCE_NAME="instance-1"
#GCP_INSTANCE_ZONE="us-east1-c"
GCP_INSTANCE_NAME="instance-2"
GCP_INSTANCE_ZONE="us-east1-d"
VM_INPUT_DIR="~/nlp/src/data/tmp"
VM_PIPELINE="~/nlp/src/pipeline/interactive_pipeline.sh"

#####FUNCTIONS

usage() {
	echo "start_modules: entry point that sets off the series of commands that start the pipeline modules remotely"
	echo -e "\nUsage: start_modules.sh [CLAIM]"
}

clean_up() {
	exit 0
}

start_pipeline() {
  gcloud compute ssh "$GCP_INSTANCE_NAME" --zone "$GCP_INSTANCE_ZONE" --command "export PYTHONPATH=/home/kikuchio/nlp/src/ && $VM_PIPELINE"
}

######MAIN

trap clean_up SIGINT SIGTERM SIGHUP

case "$1" in
	"--help")
					usage 
					exit 0
					;;
esac

start_pipeline
