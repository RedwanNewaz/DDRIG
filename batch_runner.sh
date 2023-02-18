#!/usr/bin/env bash

array=( 'N17E073' 'N43W080' 'N45W123' 'N47W124' )
NUM_SAMPLES=500

for robot in 3 4 5 6
do
	for env in "${array[@]}"
	do
	echo "[+] robot $robot executing experiment $env"
	args="--config AK/experiments/configs/ak.yaml --env-name $env  --strategy distributed --seed 10 --output-dir results/newDistributed/exp-$robot-robot-$env --max-num-samples $NUM_SAMPLES --no-viz --num-robots $robot"
	./workspace.py $args 
	done
done
