import logging
import os
import pickle
from functools import partial

import numpy as np
import pandas as pd
import torch
from src.configs.dataloader_configs import StratsDataloaderConfig, SupervisedView
from src.configs.train_configs import TrainConfig
from src.data.dataloaders.utils import (
    L2_KEYS,
    BaseDataloaderCreator,
    alt_collate_value,
    load_encounters,
    process_df_to_dict,
)
from src.utils.data_utils import Split
from tqdm.auto import tqdm


def collate_strats_triplets(seq_len, half_dtype, batch, device=None):
    collated_batch = {SupervisedView.EARLY_FUSION.value: dict()}
    lengths = []

    L1_KEYS = [SupervisedView.EARLY_FUSION.value]

    for l1 in L1_KEYS:
        for l2 in L2_KEYS:
            tensors, lengths = alt_collate_value(seq_len, batch, half_dtype, l1, l2, device)
            assert tensors.isfinite().all(), "NaN or Inf in tensor"
            tensors.to(device)
            collated_batch[l1][l2] = tensors
            collated_batch[l1]["length"] = lengths

    collated_batch["forecast_target"] = torch.utils.data.default_collate(
        [each["forecast_target"] for each in batch]
    )
    collated_batch["forecast_target_mask"] = torch.utils.data.default_collate(
        [each["forecast_target_mask"] for each in batch]
    )
    if half_dtype:
        collated_batch["forecast_target"] = collated_batch["forecast_target"].half()
        collated_batch["forecast_target_mask"] = collated_batch["forecast_target_mask"].half()
    try:
        patient_ids = [each["patient_id"] for each in batch]
        collated_batch["patient_id"] = patient_ids

        encounter_date = [each["encounter_date"] for each in batch]
        collated_batch["encounter_date"] = encounter_date

        encounter_type = [each["type"] for each in batch]
        collated_batch["type"] = encounter_type
    except:  # noqa E722
        pass

    outcome = [each["outcome"] for each in batch]
    raw_outcome = [each["raw_outcome"] for each in batch]
    if outcome[0] is None:
        collated_batch["outcome"] = None
        collated_batch["raw_outcome"] = None
    else:
        collated_batch["outcome"] = torch.as_tensor(outcome, device=device)
        collated_batch["raw_outcome"] = torch.as_tensor(raw_outcome, device=device)

    return collated_batch


class StratsDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: TrainConfig, data_dict, encounters, split):
        self.cfg = cfg
        self.split = split
        self.args: StratsDataloaderConfig = cfg.dataloader_config
        self.data_dict = data_dict
        self.encounters = encounters.reset_index(drop=True)
        self.patient_ids = list(self.encounters["patient_id"].unique())
        self.std_time = encounters.date.std()
        self.deidentified_keys = [
            "outcome",
            "raw_outcome",
            "pre",
            "forecast",
            "early_fusion",
            "post",
        ]
        # check that n_cat_value, n_cat_variable, n_variable
        # n_cat_variable: int = 52 # variables 52 and up are continuous
        # n_cat_value: int = 7761
        # n_variable: int = 3275
        if self.cfg.debug:
            n_cat_variable = 0
            n_variable = 0
            n_cat_value = 0
            for value in tqdm(
                self.data_dict.values(),
                desc="Checking n_cat_value, n_cat_variable, n_variable",
            ):
                n_cat_variable = max(value[value.is_cat.astype(bool)].variable.max(), n_cat_variable)
                n_variable = max(value.variable.max(), n_variable)
                n_cat_value = max(value[value.is_cat.astype(bool)].value.max(), n_cat_value)
            assert self.args.n_cat_variable == 1 + n_cat_variable
            assert self.args.n_variable == 1 + n_variable
            assert self.args.n_cat_value == 1 + n_cat_value

    def generate_strats_data_point(self, idx):
        """
        post_cutoff: used to make post_filtered indexes that span ~ [event, event+{post_cutoff}] days
        pre_cutoff: used to make post_filtered indexes that span ~ [event-{pre_cutoff}, event] days
        """
        patient_id = self.patient_ids[idx]
        args: StratsDataloaderConfig = self.args
        assert isinstance(args, StratsDataloaderConfig), type(args)
        # get data for the patient_id
        subset_df = self.data_dict[patient_id]
        assert subset_df.date.is_monotonic_increasing
        # make set of all forecasting windows (i.e. windows of length self.args.forecast_window days)
        encounter_date = subset_df.iloc[self.args.min_obs].date
        if pd.api.types.is_numeric_dtype(subset_df.date.dtype):
            subset_df["window_num"] = np.ceil(subset_df.date - encounter_date).astype(int)
        else:
            subset_df["window_num"] = np.ceil(
                (subset_df.date - encounter_date) / pd.to_timedelta(self.args.forecast_window, unit="d")
            ).astype(int)

        windows = [each for each in subset_df.window_num.unique() if each > 0]
        # sample a forecasting window
        if self.split == Split.TRAIN:
            forecast_window_num = np.random.choice(windows)
        else:
            rng = np.random.default_rng(idx)
            forecast_window_num = rng.choice(windows)
        # create a dataframe of the forecast window data
        forecast_window = subset_df[subset_df.window_num == forecast_window_num].copy()
        cat_forecast_window = forecast_window[forecast_window.is_cat.astype(bool)].copy()
        cont_forecast_window = forecast_window[~forecast_window.is_cat.astype(bool)].copy()
        cont_forecast_dict = {
            k: float(v.value)
            for k, v in cont_forecast_window.groupby("variable", observed=True).first().iterrows()
        }
        cat_forecast_vals = cat_forecast_window.value.unique().astype(int)
        forecast_window = (
            forecast_window.groupby("variable", observed=True).first().reset_index()
        )  # the oldest value of each variable in the window is taken (first value is the oldest)

        # forecast array:
        n_cont_var = self.cfg.dataset_config.n_variable - self.cfg.dataset_config.n_cat_variable
        keys = [k for k in cont_forecast_dict.keys()]
        indexes = torch.as_tensor(keys, dtype=torch.int64) - n_cont_var
        values = torch.as_tensor([cont_forecast_dict[k] for k in keys])
        cont_forecast_tensor = torch.zeros(n_cont_var)
        cont_forecast_mask = torch.zeros(n_cont_var)
        cont_forecast_tensor[indexes] = values
        cont_forecast_mask[indexes] = 1

        cat_forecast_tensor = torch.zeros(self.cfg.dataset_config.n_cat_value)
        if cat_forecast_vals.sum() > 0:
            cat_forecast_tensor[cat_forecast_vals] = 1
        # no masking for categoricals as it is one hot encoded
        cat_forecast_mask = torch.ones(self.cfg.dataset_config.n_cat_value)

        forecast_target = torch.concat((cont_forecast_tensor, cat_forecast_tensor), dim=0)
        forecast_target_mask = torch.concat((cont_forecast_mask, cat_forecast_mask), dim=0)
        assert forecast_target.shape == forecast_target_mask.shape
        assert forecast_target.shape[0] == n_cont_var + self.cfg.dataset_config.n_cat_value
        assert forecast_target_mask.sum() == self.cfg.dataset_config.n_cat_value + len(
            cont_forecast_dict.keys()
        )

        # create a dataframe of the pre forecast window data
        pre_window = subset_df[subset_df.window_num < forecast_window_num]
        # get the most recent obsevations to the forecast if we have to remove datapoints
        num_pre_samples = min(self.args.max_obs, pre_window.shape[0])
        pre_window = pre_window.tail(num_pre_samples)
        encounter_date = pre_window.date.iloc[int(len(pre_window) / 2) - 1]  # int(len(pre_window) / 2) - 1

        obs = dict(
            patient_id=patient_id,
            encounter_date=encounter_date,
            forecast_target=forecast_target,
            forecast_target_mask=forecast_target_mask,
        )
        obs[SupervisedView.EARLY_FUSION.value] = process_df_to_dict(pre_window, encounter_date, self.std_time)

        obs["outcome"] = None
        obs["raw_outcome"] = None
        obs["type"] = None
        return obs

    def get_encounter(self, idx):
        return self.generate_strats_data_point(idx)

    def __getitem__(self, idx):
        return self.get_encounter(idx)

    def __len__(self):
        return len(self.patient_ids)


class StratsDataLoaderCreator(BaseDataloaderCreator):
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.args: StratsDataloaderConfig = self.cfg.dataloader_config
        logging.info("Loading Timeseries Data")
        with open(os.path.join(cfg.dataset_config.data_dir, "timeseries_dict.pkl"), "rb") as f:
            try:
                self.data_dict = pickle.load(f)
            except ModuleNotFoundError:
                self.data_dict = pd.compat.pickle_compat.load(f)
                print("Loading data via pd.compat!")

    def get_dataset(self, encounter_set, split):
        # Load and shuffle encounters
        frac = 1
        # Fixed on Sept. 17 - subsample validation (not test set)
        if split == Split.TRAIN or split == Split.VAL:
            assert self.args.subsample <= 1 and self.args.subsample > 0, "subsample should be between 0 and 1"
            frac = self.args.subsample

        encounters = load_encounters(self.cfg, encounter_set, split, self.args.inpatient_only).sample(
            frac=frac, random_state=self.args.seed
        )
        logging.info(f"len(encounters) for split {split.value}: {len(encounters)}")

        dataset = StratsDataset(self.cfg, self.data_dict, encounters, split)
        return dataset

    def _get_dataloader(self, dataset, split):
        drop_last = split != Split.TEST
        shuffle = split == Split.TRAIN
        seq_len = self.cfg.model_config.seq_len if self.cfg.model_config.pad_seq else None
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.args.num_cpus,
            pin_memory=True,
            collate_fn=partial(collate_strats_triplets, seq_len, self.args.half_dtype),
            prefetch_factor=(None if self.args.num_cpus == 0 else self.args.prefetch_factor),
        )

    def get_dataloaders(self):
        train_dataset = self.get_dataset(self.args.encounter_set, Split.TRAIN)
        test_dataset = self.get_dataset(self.args.encounter_set, Split.TEST)
        val_dataset = self.get_dataset(self.args.encounter_set, Split.VAL)

        train_dataloader = self._get_dataloader(train_dataset, split=Split.TRAIN)
        test_dataloader = self._get_dataloader(test_dataset, split=Split.TEST)
        val_dataloader = self._get_dataloader(val_dataset, split=Split.VAL)
        return train_dataloader, val_dataloader, test_dataloader


def push_to(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        device_dict = dict()
        for k, v in batch.items():
            device_dict[k] = push_to(v, device)
        return device_dict
    else:
        return batch
