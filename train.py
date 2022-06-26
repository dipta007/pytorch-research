import logging

import hydra
import omegaconf
import torch

import wandb
from utils.utils import (
    DotDict,
    print_config,
    print_repo_info,
    seed_everything,
    wandb_log_code,
)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="root")
def main(cfg):
    config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config = DotDict(config)
    config.cuda = config.cuda and (
        torch.cuda.is_available() or torch.backends.mps.is_available()
    )
    config.device = (
        ("cuda" if torch.cuda.is_available() else "mps") if config.cuda else "cpu"
    )
    if not config.debug:
        run = wandb.init(
            name=cfg.exp_name,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            config=config,
            settings=wandb.Settings(start_method="thread"),
        )
        wandb_log_code(run)
        config = DotDict(wandb.config)

    print_repo_info()
    print_config(config)

    log.info(f"Running on {config.device}")
    seed_everything(config.seed)

    trainer = hydra.utils.instantiate(config.train.trainer)(config=config)
    trainer.run()


if __name__ == "__main__":
    main()
