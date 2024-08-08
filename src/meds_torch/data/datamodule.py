from typing import Any

from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from meds_torch.data.components.multiwindow_pytorch_dataset import (
    MultiWindowPytorchDataset,
)
from meds_torch.data.components.pytorch_dataset import PytorchDataset
from meds_torch.utils.module_class import Module


def get_dataset(cfg: DictConfig, split) -> PytorchDataset:
    if cfg.name == "multiwindow_pytorch_dataset":
        return MultiWindowPytorchDataset(cfg, split)
    elif cfg.name == "pytorch_dataset":
        return PytorchDataset(cfg, split)
    else:
        raise NotImplementedError(f"{cfg.name} not implemented!")


class MEDSDataModule(LightningDataModule, Module):
    """`LightningDataModule` for the MEDS pytorch dataset.

    TODO: Add documentation

    A `LightningDataModule` implements 7 key methods:

    ```python     def prepare_data(self):     # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).     #
    Pre-process and loading data, save to disk, etc...

    def setup(self, stage): # Things to do on every process in DDP. # Load data, set variables, etc...

    def train_dataloader(self): # return train dataloader

    def val_dataloader(self): # return validation dataloader

    def test_dataloader(self): # return test dataloader

    def predict_dataloader(self): # return predict dataloader

    def teardown(self, stage):     # Called on every process in DDP.     # Clean up after fit or test. ```

    This allows you to share a full dataset without explaining how to download, split, transform and process
    the data.

    Read the docs:
    https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        cfg: DictConfig = None,
    ) -> None:
        """Initialize a `MEDSDataModule`."""
        super().__init__()
        self.cfg = cfg

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only within a
        single process on CPU, so you can safely add your downloading logic within. In case of multi-node
        training, the execution of this hook depends upon `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called
        after `self.prepare_data()` and there is a barrier in between which ensures that all the processes
        proceed to `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to
            ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.cfg.dataloader.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.cfg.dataloader.batch_size}) is not divisible by "
                    f"the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.cfg.dataloader.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        self.data_train = get_dataset(self.cfg, split="train")
        if stage != "train":  # TODO: remove this after we have more test data
            self.data_val = get_dataset(self.cfg, split="tuning")
        if stage in ["test", None]:
            self.data_test = get_dataset(self.cfg, split="held_out")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            shuffle=True,
            collate_fn=self.data_train.collate,
            drop_last=True,
            **self.cfg.dataloader,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            collate_fn=self.data_val.collate,
            **self.cfg.dataloader,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            collate_fn=self.data_test.collate,
            **self.cfg.dataloader,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """


if __name__ == "__main__":
    _ = MEDSDataModule()
