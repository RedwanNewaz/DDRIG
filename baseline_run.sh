#!/usr/bin/env bash 








array=( 'N17E073' 'N43W080' 'N45W123' 'N47W124' )
for env in "${array[@]}"
do
	echo "[+] executing experiment $env"
	./main.py --config AK/experiments/configs/ak.yaml --env-name $env  --strategy myopic --seed 0 --output-dir results/single/exp_ --max-num-samples 700 --no-viz
done
