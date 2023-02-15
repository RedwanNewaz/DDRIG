# PyPolo++

## How to configure 
create a conda environment and install all the requirements 
```bash 
conda create -n rig python=3.8
conda activate rig 
pip install -r requirements.txt 
```

### How to save experiment results
use appropriate arguments to save results, e.g.,
```bash 
python workspace.py --config AK/experiments/configs/ak.yaml --env-name N45W123 --strategy distributed --seed 0 --save-fig results/exp8 --max-num-samples 200 --no-viz
```


### How to use it in PyCharm 
use the following argument in pycharm `` Run > edit configuration ``
```bash 
--config AK/experiments/configs/ak.yaml --env-name N45W123 --strategy distributed --seed 0
```

