## References:
1. [Wandb with Hydra](https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw)
2. [Weight & Bias Tutorial](https://theaisummer.com/weights-and-biases-tutorial/)

## Used Packages
1. Pytorch
2. Wandb
3. Hydra
4. TQDM
5. pre-commit
   1. black
   2. isort

## Need to do:
- [] Edit `sweep.yaml` to change the hyperparameters
- [] tally parameters trainable and non-trainable
- [] Resume run from checkpoint with wandb integration
- [] Add logging to each step
- [] List of schedulers to trainer

## Sweep on SLURM
1. Edit `sweep.yaml` file accordingly
2. Run `wandb sweep sweep.yaml`
3. Get the sweep ID
4. Run `./ada.sh wandb agent $SWEEP_ID` as many times as u want
