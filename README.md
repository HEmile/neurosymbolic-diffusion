## Setup
Install `uv`, then run `uv sync`. 

## Running experiments
For all scripts below, the hyperparameters as reported in the paper should be used. If not, please add an issue.
This project relies on wandb for reporting measures. Some customisation may be needed to ensure runs go in the right wandb project. 

### Commands
MNIST Add N=4:
```
uv run expressive/experiments/mnist_op/mnistop.py
```

MNIST Add N=15:
```
uv run expressive/experiments/mnist_op/mnistop.py --N 15 --epochs 1000
```

Path Planning 12x12:
```
./expressive/experiments/path_planning/download.sh # If data is not yet downloaded
uv run expressive/experiments/path_planning/path_planning.py
```

Path Planning 30x30:
```
./expressive/experiments/path_planning/download.sh  # If data is not yet downloaded
uv run expressive/experiments/path_planning/data/merge.py # Data postprocessing step required for N=30
uv run expressive/experiments/path_planning/path_planning.py --grid_size 30 --loss_S 2 --variational_K 2 --test_K 2
```

MNIST Half: 
```
cd expressive/experiments/rsbench
uv run nesydiffusion.py
```

MNIST Even/Odd
```
cd expressive/experiments/rsbench
uv run nesydiffusion.py --dataset shortmnist
```

BDD-OIA: We use preprocessed embeddings. Download these from the RSBench data at https://drive.google.com/drive/folders/1PB4FZrZ_iZ_XH28u-nAykkVqMLDYqACB . Grab `BDD-OIA-preprocessed.zip`, and extract in `expressive/experiments/rsbench/data/`. 
```
cd expressive/experiments/rsbench
uv run nesydiffusion.py --dataset boia --task boia --lr 0.0001 --batch_size 256 --epochs 30 --w_denoise_weight 0.000005 --entropy_weight 2.0 --backbone fullentangled
```
