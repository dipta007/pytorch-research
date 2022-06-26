# Steps to start working
1. Add data on `data` folder
2. Add dataset generator on `dataloader` folder
3. Add models on `model` folder
4. Add trainer on `trainer` folder
   1. Implement `__init__`
   2. Implement `get_loss`
   3. Implement `run_metrics` if needed
   4. Implement `get_optimizer` if needed
5. Add config to `config` folder
6. Add tests class on `inference` folder
7. Run tests from `test.py`

## References:
1. [Wandb with Hydra](https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw)
2. [Weight & Bias Tutorial](https://theaisummer.com/weights-and-biases-tutorial/)
3. [Sweep Configs](https://docs.wandb.ai/guides/sweeps/configuration)

## Supported
1. Fully Pytorch compatible
2. Wandb integration
3. Hydra integration
4. TQDM style progress bars
5. pre-commit for commit hooks for fixed code styling
   1. black
   2. isort
6. CUDA, cpu or Apple mps support
7. wandb sweep for hyperparameter finding
8.

## Need to do:
- [ ] Edit `sweep.yaml` to change the hyperparameters
- [x] tally parameters trainable and non-trainable
- [ ] Resume run from checkpoint with wandb integration
- [x] Add logging.info() to each step
- [ ] List of schedulers to trainer
- [ ] Add GPU memory profiling
- [ ] Add example

## Sweep on SLURM
1. Edit `sweep.yaml` file accordingly
2. Run `wandb sweep sweep.yaml`
3. Get the sweep ID
4. Run `./ada.sh wandb agent $SWEEP_ID` / `./ada.sh wandb agent --count 1 $SWEEP_ID` as many times as u want

## Necessary Cmds:
1. `srun --mem=40000 --time=24:00:00 --gres=gpu:1 --pty --constraint=rtx_8000 bash`
