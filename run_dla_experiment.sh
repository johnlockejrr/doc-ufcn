#!/bin/bash

TMP_DIR="./tmp"
slack=true

helpFunction()
{
   echo ""
   echo "Usage: $0 -c config -s bool"
   echo -e "\t-c Path to csv configurations file"
   echo -e "\t-s Send slack notifications"
   exit 1 # Exit script after printing help
}

while getopts "c:s:" opt
do
   case "$opt" in
      c ) config="$OPTARG" ;;
      s ) slack="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$config" ]
then
   echo "Configuration file required";
   helpFunction
fi

if [ -z $STY ]
then
    exec screen -dm -S dla-training /bin/bash "$0"
fi

python3 retrieve_experiments_configs.py --config "$config"

index=1

for filename in ${TMP_DIR}/*; do

    echo
    echo "Running experiment $index with $filename"

    if [ "$slack" == "false" ] || [ "$slack" == "False" ]
    then
        python3 run_experiment.py with experiments_config.json "$filename" 2>&1 | tee >(grep --line-buffered -v "(prog)" > DLA_train_"${index}".log)
    else
        python3 run_experiment.py with experiments_config.json "$filename" 2>&1 | tee >(grep --line-buffered -v "(prog)" > DLA_train_"${index}".log) \
        && (python3 notify-slack.py "INFO: Experiment completed" --log_file DLA_train_"${index}".log) \
        || (python3 notify-slack.py "ERROR: Experiment failed" --log_file DLA_train_"${index}".log ; exit)
    fi
    
    index=$((index+1))

done

echo 
echo "Experiments done!"

rm -r $TMP_DIR



