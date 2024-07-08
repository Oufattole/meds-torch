import logging
import os
import pickle
from functools import partial

import numpy as np
import pandas as pd
import torch
from src.configs.dataloader_configs import OCPDataloaderConfig, SupervisedView
from src.configs.train_configs import TrainConfig
from src.data.dataloaders.utils import (
    L2_KEYS,
    BaseDataloaderCreator,
    alt_collate_value,
    load_encounters,
    process_df_to_dict,
)
from src.utils.data_utils import Split


def collate_ocp_triplets(seq_len, half_dtype, batch, device=None):
    collated_batch = dict(early_fusion=dict())

    L1_KEYS = [SupervisedView.EARLY_FUSION.value]
    lengths = []

    for l1 in L1_KEYS:
        for l2 in L2_KEYS:
            # import pdb; pdb.set_trace() # [len(data["early_fusion"]["cat"]["date"]) for data in batch]
            tensors, lengths = alt_collate_value(seq_len, batch, half_dtype, l1, l2, device)
            tensors.to(device)
            collated_batch[l1][l2] = tensors
            collated_batch[l1]["length"] = lengths
    patient_ids = [each["patient_id"] for each in batch]
    flip = [each["flip"] for each in batch]
    collated_batch["patient_id"] = patient_ids
    collated_batch["flip"] = torch.as_tensor(flip, device=device)

    # encounter_date = [each["encounter_date"] for each in batch]
    # collated_batch["encounter_date"] = encounter_date

    return collated_batch


class OCPDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: TrainConfig, data_dict, encounters, split: Split):
        self.cfg = cfg
        self.args: OCPDataloaderConfig = cfg.dataloader_config
        self.data_dict = data_dict
        self.split = split
        self.patient_ids = []

        self.patient_ids = np.array(encounters.patient_id.unique())
        # shuffle the mrns using data_seed
        np.random.default_rng(seed=self.args.seed).shuffle(self.patient_ids)

        self.std_time = encounters.date.std()

    def generate_ocp_data(self, idx, swap):
        """
        left_cutoff: used to make left cutoff of dataset of args.max_obs size dataset
        right_cutoff: used to make right cutoff of dataset of args.max_obs size dataset
        """
        args = self.args
        std_time = self.std_time
        patient_id = self.patient_ids[idx]
        subset_df = self.data_dict[patient_id].copy()
        assert subset_df.date.is_monotonic_increasing

        assert len(subset_df) >= 2 * args.min_obs, f"ocp sample has only {len(subset_df)} samples!"

        if len(subset_df) <= args.max_obs:
            left = 0
        else:
            if self.split == Split.TRAIN:
                left = np.random.randint(low=0, high=len(subset_df) - args.max_obs)
            else:
                # use constant seed for validation and test set
                constant_rng = np.random.default_rng(seed=idx)
                left = constant_rng.integers(low=0, high=len(subset_df) - args.max_obs)

        # Sequence must be even length, so pre and post are the same size
        right = min(left + args.max_obs, left + len(subset_df) - len(subset_df) % 2)
        cutoff = int((left + right) / 2)

        pre_filtered_rows = subset_df.iloc[left:cutoff].reset_index(drop=True)
        post_filtered_rows = subset_df.iloc[cutoff:right].reset_index(drop=True)

        gap = post_filtered_rows["date"].min() - pre_filtered_rows["date"].max()
        if swap:
            date_add = post_filtered_rows["date"].max() - pre_filtered_rows["date"].min()
            pre_filtered_rows["date"] += date_add + gap
            enc_date = post_filtered_rows["date"].max()  # Post max
            filtered_rows = pd.concat([post_filtered_rows, pre_filtered_rows], axis=0)
            assert enc_date <= pre_filtered_rows["date"].min()
        else:
            filtered_rows = pd.concat([pre_filtered_rows, post_filtered_rows], axis=0, ignore_index=True)
            enc_date = pre_filtered_rows["date"].max()  # Pre max
            assert enc_date <= post_filtered_rows["date"].min()
        filtered_rows = filtered_rows.sort_values(by="date").reset_index(drop=True)
        assert filtered_rows.date.is_monotonic_increasing

        obs = dict(
            patient_id=patient_id,
            encounter_date=enc_date,
            flip=swap,
        )
        obs[SupervisedView.EARLY_FUSION.value] = process_df_to_dict(filtered_rows, enc_date, std_time)

        return obs

    def get_encounter(self, idx, swap):
        return self.generate_ocp_data(idx, swap)

    def __getitem__(self, idx):
        # swap: label whether the sequence is swapped
        if self.split == Split.TRAIN:
            swap = np.random.choice([0, 1])
        else:
            # use constant seed for validation and test set
            swap = idx % 2

        if isinstance(idx, int):
            return self.get_encounter(idx, swap)
        else:
            raise NotImplementedError("Only int indexing is supported")
            return [self.get_encounter(i, swap) for i in idx]

    def __len__(self):
        return len(self.patient_ids)


class OCPDataLoaderCreator(BaseDataloaderCreator):
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.args = cfg.dataloader_config
        logging.info("Loading Timeseries Data")

        with open(os.path.join(cfg.dataset_config.data_dir, "timeseries_dict.pkl"), "rb") as f:
            try:
                self.data_dict = pickle.load(f)
            except ModuleNotFoundError:
                self.data_dict = pd.compat.pickle_compat.load(f)
                print("Loading data via pd.compat!")

    def get_dataset(self, encounter_set, split):
        frac = 1
        encounters = load_encounters(self.cfg, encounter_set, split, self.args.inpatient_only).sample(
            frac=frac, random_state=self.args.seed
        )
        logging.info(f"len(encounters) for split {split.value}: {len(encounters)}")
        dataset = OCPDataset(self.cfg, self.data_dict, encounters, split)

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
            collate_fn=partial(collate_ocp_triplets, seq_len, self.args.half_dtype),
            prefetch_factor=(None if self.args.num_cpus == 0 else self.args.prefetch_factor),
        )
