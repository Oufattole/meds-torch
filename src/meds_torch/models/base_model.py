import lightning as L
import torch
from omegaconf import DictConfig

from meds_torch.utils.module_class import Module


class BaseModule(L.LightningModule, Module):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        """Initializes the BaseModule with the given configuration, setting up various components such as the
        optimizer, scheduler, model, and input encoder.

        Parameters: cfg (DictConfig): The configuration dictionary specifying the setup of the module,
        including required elements like the task name and optional elements                   for configuring
        the optimizer, scheduler, model, and input encoder.

        Raises: ValueError: If the task name is not specified in the configuration.

        Returns: None
        """
        super().__init__()
        self.cfg = cfg
        # shared components
        self.optimizer = cfg.optimizer
        self.scheduler = cfg.scheduler
        self.model = cfg.backbone
        self.input_encoder = cfg.input_encoder

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        return optimizer

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate, test, or
        predict.

        This is a good hook when you need to build models dynamically or adjust something about them. This
        hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.cfg.compile and stage == "fit":
            self.net = torch.compile(self.net)
