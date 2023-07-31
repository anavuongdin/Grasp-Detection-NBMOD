#!/bin/bash

# Check if the batch number is provided as a command-line argument
if [ -z "$1" ]; then
    echo "Usage: $0 <batch_number>"
    exit 1
fi

# Get the batch number from the command-line argument
INPUT_BATCH="$1"

# Replace occurrences of "batch_7" with the user input stored in "INPUT_BATCH"
CMD1="rm -rf data/grasp-anything-backup/"
CMD2="cp ../../../lthieu/data/batch_${INPUT_BATCH}.pkl data"
CMD3="cp -r ../robotic-grasping/data/grasp-anything/ grasp-anything-backup"
CMD4="mv grasp-anything-backup/ data/"
CMD5="python setup_pipe.py data/batch_${INPUT_BATCH}.pkl ../robotic-grasping/data/grasp-anything/"

# Execute the commands
$CMD1
$CMD2
$CMD3
$CMD4
$CMD5
