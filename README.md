# Steps to start working
1. Run `./install.sh` to install dependencies
2. Add data on `data` folder
3. Add dataset generator on `dataloader` folder
4. Add models on `model` folder
5. Add trainer on `trainer` folder
   1. Implement `__init__`
   2. Implement `get_loss`
   3. Implement `run_metrics` if needed
   4. Implement `get_optimizer` if needed
   5. Implement `get_schedulers` if needed
6. Add config to `config` folder
7. Run training using `python train.py +experiment=$EXP_NAME`
8. Add tests class on `inference` folder
9. Run tests using `python test.py --exp_name=$EXP_NAME --test_type=$TEST_TYPE`

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
8. PyTorch scheduler support

## To do:
- [x] Edit `sweep.yaml` to change the hyperparameters
- [x] tally parameters trainable and non-trainable
- [ ] Resume run from checkpoint with wandb integration
- [x] Add logging.info() to each step
- [x] Schedulers to trainer
- [ ] Add GPU memory profiling
- [ ] Add example

## Sweep on SLURM
1. Edit `sweep.yaml` file accordingly
2. Run `wandb sweep sweep.yaml`
3. Get the sweep ID
4. Run `./ada.sh wandb agent $SWEEP_ID` / `./ada.sh wandb agent --count 1 $SWEEP_ID` as many times as u want

## Necessary Cmds:
1. `srun --mem=40000 --time=24:00:00 --gres=gpu:1 --pty --constraint=rtx_8000 bash`
