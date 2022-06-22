import json
import logging

from base.base_trainer import BaseTrainer

log = logging.getLogger(__name__)


class NaaclTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer):
        self.optimizer_partial = optimizer
        super().__init__(cfg, model)

        log.info(json.dumps(cfg, indent=4))

    def run(self):
        self.fit()

    def get_optimizer(self):
        return self.optimizer_partial(self.model.parameters())
