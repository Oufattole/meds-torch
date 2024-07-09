import json
import os
from io import StringIO
from pathlib import Path

import pyarrow
import pyarrow.parquet
import meds
import tempfile
import datasets
import polars as pl
import datetime
import polars as pl
from hydra import compose, initialize
import femr.index
import femr.splits
import os
import femr.models.tokenizer
import femr.models.processor
import femr.models.tasks

from MEDS_torch.clmbr.clmbr_model import CLMBR
from MEDS_torch.utils import list_subdir_files



def test_clmbr(tmp_path):
    input_meds = pl.read_parquet("tests/processed/final_cohort/**/*").rename(dict(timestamp="time", numerical_value="numeric_value"))
    # impute time for static data with oldest time -- femr doesn't seem to support static data without this
    input_meds = input_meds.group_by("patient_id").agg([pl.col("time").fill_null(pl.col("time").min()), "code", "numeric_value"]).explode("time", "code", "numeric_value")
    # add a birth date
    fake_bdays = input_meds.group_by("patient_id").agg(pl.col("time").min())
    fake_bdays = fake_bdays.with_columns(pl.lit(meds.birth_code).alias("code"), pl.lit(None).alias("numeric_value"))
    input_meds = pl.concat([input_meds, fake_bdays])


    def unnest_df(df):
        return df.explode("events").unnest("events").explode("measurements").unnest("measurements")

    def nest(df: pl.DataFrame)-> pl.DataFrame:
        """An older version of meds used this nested structure
        
        This function reconstructs the nested structure by grouping by patient ids and subnesting by event times

        Args:
            df (pl.DataFrame): 

        Returns:
            pl.DataFrame: double nested meds dataframe
        """
        # Step 1: Group by patient_id and time, then aggregate the relevant fields into structs
        nested_measurements = df.group_by(["patient_id", "time"]).agg([
            pl.struct(pl.exclude(["patient_id", "time"])).alias("measurements")
        ])

        # Step 2: Group by patient_id and aggregate the structs into lists
        nested_events = nested_measurements.group_by("patient_id").agg([
            pl.struct(["time", "measurements"]).alias("events")
        ])
        return nested_events

    def convert_meds_to_femr_meds(df):
        # confirm collumns are allows
        allowed_columns = {'patient_id', 'time', 'code', 'text_value', 'numeric_value', 'datetime_value', 'metadata'}
        assert all([each in allowed_columns for each in df.columns])
        return nest(df)
    df = convert_meds_to_femr_meds(input_meds)

    patient_schema = meds.patient_schema()
    patient_table = pyarrow.Table.from_pylist(df.to_dicts(), patient_schema)
    parquet_path = f"{tmp_path}/test.parquet"
    pyarrow.parquet.write_table(patient_table, parquet_path)
    dataset = datasets.Dataset.from_parquet(f"{tmp_path}/*")
    
    
    index = femr.index.PatientIndex(dataset, num_proc=4)
    main_split = femr.splits.generate_hash_split(index.get_patient_ids(), 97, frac_test=0.15)


    os.mkdir(os.path.join(tmp_path, 'clmbr_model'))
    # Note that we want to save this to the target directory since this is important information

    main_split.save_to_csv(os.path.join(tmp_path, "clmbr_model", "main_split.csv"))

    train_split = femr.splits.generate_hash_split(main_split.train_patient_ids, 87, frac_test=0.15)

    print(train_split.train_patient_ids)
    print(train_split.test_patient_ids)

    main_dataset = main_split.split_dataset(dataset, index)
    train_dataset = train_split.split_dataset(main_dataset['train'], femr.index.PatientIndex(main_dataset['train'], num_proc=4))

    print(train_dataset)
    # First, we need to train a tokenizer
    # NOTE: A vocab size of 128 is probably too low for a real model. 128 was chosen to make this tutorial quick to run
    tokenizer = femr.models.tokenizer.train_tokenizer(
        main_dataset['train'], vocab_size=128, num_proc=4)

    # Save the tokenizer to the same directory as the model
    tokenizer.save_pretrained(os.path.join(tmp_path, "clmbr_model"))



    # Second, we need to create batches. We define the CLMBR task at this time

    clmbr_task = femr.models.tasks.CLMBRTask(clmbr_vocab_size=64)

    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, clmbr_task)

    # We can do this one patient at a time
    example_batch = processor.collate([processor.convert_patient(train_dataset['train'][0], tensor_type='pt')])

    # # But generally we want to convert entire datasets
    # train_batches = processor.convert_dataset(train_dataset, tokens_per_batch=2, num_proc=1)

    # # Convert our batches to pytorch tensors
    # train_batches.set_format("pt")