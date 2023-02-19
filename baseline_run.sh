#!/usr/bin/env bash 

SAMPLES=500


run()
{
	mkdir -p $1
	array=( 'N17E073' 'N43W080' 'N45W123' 'N47W124' )
	for env in "${array[@]}"
	do
		echo "[+] executing experiment $env"
		./main.py --config AK/experiments/configs/ak.yaml --env-name $env  --strategy myopic --seed 0 --output-dir $1 --max-num-samples $SAMPLES --no-viz
	done
}


run "results/single5"
