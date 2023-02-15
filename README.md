# PyPolo++

## How to run 
use the following argument in pycharm `` Run > edit configuration ``
```bash 
--config AK/experiments/configs/ak.yaml --env-name N45W123 --strategy distributed --seed 0

```

to save experiment results
```bash 
./workspace.py --config AK/experiments/configs/ak.yaml --env-name N45W123 --strategy distributed --seed 0 --save-fig results/exp8 --max-num-samples 200 --no-viz

```