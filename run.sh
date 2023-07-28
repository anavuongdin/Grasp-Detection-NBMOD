#!/bin/bash

# Read the input from the user and store it in the variable "INPUT_BATCH"
read -p "Enter the batch number (e.g., 7): " INPUT_BATCH

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
