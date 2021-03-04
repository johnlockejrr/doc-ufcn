#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -o -s"
   echo -e "\t-s Send slack notifications"
   exit 1 # Exit script after printing help
}

slack=true

while getopts "o:s:" opt
do
   case "$opt" in
      s ) slack="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

if [ -z $STY ]
then
    exec screen -dm -S dla-training /bin/bash "$0"
fi

if [ "$slack" == "false" ] || [ "$slack" == "False" ]
then
    python3 run_experiment.py with config.json 2>&1 | tee >(grep --line-buffered -v "(prog)" > DLA_train.log)
else
    python3 run_experiment.py with config.json 2>&1 | tee >(grep --line-buffered -v "(prog)" > DLA_train.log) \
    && python3 notify-slack.py "INFO: Experiment completed" \
    || (python3 notify-slack.py "ERROR: Experiment failed"; exit)
fi
