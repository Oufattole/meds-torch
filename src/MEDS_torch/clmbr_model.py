from pathlib import Path
from MEDS_torch.pytorch_dataset import PytorchDataset
from MEDS_torch.utils import list_subdir_files
from omegaconf import OmegaConf
from hydra import initialize, compose
from io import StringIO
import polars as pl
import json
import os
import datasets
import femr.index
import femr.splits
import sys
sys.path.append('/home/aleksia/meds-torch/tests')
import test_clmbr

# DATASET LOAD & SPLIT

# def meds_dataset(path, split):
#     MEDS_cohort_dir = path / "meds" / "final_cohort"
#     describe_codes_config = {
#         "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
#         "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
#         "max_seq_len": 512,
#     }

#     with initialize(
#         version_base=None, config_path="../src/MEDS_torch/configs"
#     ):  # path to config.yaml
#         overrides = [f"{k}={v}" for k, v in describe_codes_config.items()]
#         cfg = compose(config_name="pytorch_dataset", overrides=overrides)  # config.yaml

#     # Create the directories
#     MEDS_cohort_dir.mkdir(parents=True, exist_ok=True)
    
#     # Check the files are not empty
#     meds_files = list_subdir_files(Path(cfg.MEDS_cohort_dir), "parquet")
#     assert (
#         len(list_subdir_files(Path(cfg.MEDS_cohort_dir).parent, "parquet")) > 0
#     ), "MEDS train split Data Files Should not be Empty!"
#     for f in meds_files:
#         assert pl.read_parquet(f).shape[0] > 0, "MEDS Data Tabular Dataframe Should not be Empty!"
        
#     pyd = PytorchDataset(cfg, split)
    
#     return pyd

TARGET_DIR = os.getenv('CLMBR_DIR')

path_string = "/home/aleksia/clmbr/eicu"
path_obj = Path(path_string)

MEDS_cohort_dir = path_obj / "processed" / "final_cohort"
describe_codes_config = {
    "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
    "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
    "max_seq_len": 512,
}

with initialize(
    version_base=None, config_path="./configs"
):  # path to config.yaml
    overrides = [f"{k}={v}" for k, v in describe_codes_config.items()]
    cfg = compose(config_name="pytorch_dataset", overrides=overrides)  # config.yaml
    
train_dataset = PytorchDataset(cfg, split="train")


# TOKENIZER

import transformers
import femr.models.transformer

tokenizer = femr.models.tokenizer.train_tokenizer(
    train_dataset, vocab_size=128, num_proc=4)

# Save the tokenizer to the same directory as the model
tokenizer.save_pretrained(os.path.join(TARGET_DIR, "clmbr_model"))


# CREATE BATCHES

clmbr_task = femr.models.tasks.CLMBRTask(clmbr_vocab_size=64) # TODO (number of codes for task)

processor = femr.models.processor.FEMRBatchProcessor(tokenizer, clmbr_task)

# Convert entire dataset
train_batches = processor.convert_dataset(train_dataset, tokens_per_batch=32, num_proc=4)

# Convert batches to pytorch tensors
train_batches.set_format("pt")


# TRAIN MODEL

transformer_config = femr.models.config.FEMRTransformerConfig(
    vocab_size=tokenizer.vocab_size, 
    is_hierarchical=tokenizer.is_hierarchical, 
    n_layers=2,
    hidden_size=64, 
    intermediate_size=64*2,
    n_heads=8,
)

config = femr.models.config.FEMRModelConfig.from_transformer_task_configs(transformer_config, clmbr_task.get_task_config())

model = femr.models.transformer.FEMRModel(config)

trainer_config = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    output_dir='tmp_trainer',
    remove_unused_columns=False,
    num_train_epochs=20,

    eval_steps=20,
    evaluation_strategy="steps",

    logging_steps=20,
    logging_strategy='steps',

    prediction_loss_only=True,
)

trainer = transformers.Trainer(
    model=model,
    data_collator=processor.collate,
    train_dataset=train_batches['train'],
    eval_dataset=train_batches['test'],
    args=trainer_config,
)

trainer.train()

model.save_pretrained(os.path.join(TARGET_DIR, 'clmbr_model'))
