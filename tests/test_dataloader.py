from pathlib import Path
from MEDS_torch.pytorch_dataset import PytorchDataset
from MEDS_torch.utils import list_subdir_files
from omegaconf import OmegaConf
import tempfile
from hydra import initialize, compose
import polars as pl
import json
from io import StringIO
import os

SPLITS_JSON = """{"train/0": [239684, 1195293], "train/1": [68729, 814703], "tuning/0": [754281], "held_out/0": [1500733]}"""  # noqa: E501

MEDS_TRAIN_0 = """
patient_id,code,timestamp,numerical_value
239684,HEIGHT,,175.271115221764
239684,EYE_COLOR//BROWN,,
239684,DOB,1980-12-28T00:00:00.000000,
239684,TEMP,2010-05-11T17:41:51.000000,96.0
239684,ADMISSION//CARDIAC,2010-05-11T17:41:51.000000,
239684,HR,2010-05-11T17:41:51.000000,102.6
239684,TEMP,2010-05-11T17:48:48.000000,96.2
239684,HR,2010-05-11T17:48:48.000000,105.1
239684,TEMP,2010-05-11T18:25:35.000000,95.8
239684,HR,2010-05-11T18:25:35.000000,113.4
239684,HR,2010-05-11T18:57:18.000000,112.6
239684,TEMP,2010-05-11T18:57:18.000000,95.5
239684,DISCHARGE,2010-05-11T19:27:19.000000,
1195293,HEIGHT,,164.6868838269085
1195293,EYE_COLOR//BLUE,,
1195293,DOB,1978-06-20T00:00:00.000000,
1195293,TEMP,2010-06-20T19:23:52.000000,100.0
1195293,ADMISSION//CARDIAC,2010-06-20T19:23:52.000000,
1195293,HR,2010-06-20T19:23:52.000000,109.0
1195293,TEMP,2010-06-20T19:25:32.000000,100.0
1195293,HR,2010-06-20T19:25:32.000000,114.1
1195293,HR,2010-06-20T19:45:19.000000,119.8
1195293,TEMP,2010-06-20T19:45:19.000000,99.9
1195293,HR,2010-06-20T20:12:31.000000,112.5
1195293,TEMP,2010-06-20T20:12:31.000000,99.8
1195293,HR,2010-06-20T20:24:44.000000,107.7
1195293,TEMP,2010-06-20T20:24:44.000000,100.0
1195293,TEMP,2010-06-20T20:41:33.000000,100.4
1195293,HR,2010-06-20T20:41:33.000000,107.5
1195293,DISCHARGE,2010-06-20T20:50:04.000000,
"""
MEDS_TRAIN_1 = """
patient_id,code,timestamp,numerical_value
68729,EYE_COLOR//HAZEL,,
68729,HEIGHT,,160.3953106166676
68729,DOB,1978-03-09T00:00:00.000000,
68729,HR,2010-05-26T02:30:56.000000,86.0
68729,ADMISSION//PULMONARY,2010-05-26T02:30:56.000000,
68729,TEMP,2010-05-26T02:30:56.000000,97.8
68729,DISCHARGE,2010-05-26T04:51:52.000000,
814703,EYE_COLOR//HAZEL,,
814703,HEIGHT,,156.48559093209357
814703,DOB,1976-03-28T00:00:00.000000,
814703,TEMP,2010-02-05T05:55:39.000000,100.1
814703,HR,2010-02-05T05:55:39.000000,170.2
814703,ADMISSION//ORTHOPEDIC,2010-02-05T05:55:39.000000,
814703,DISCHARGE,2010-02-05T07:02:30.000000,
"""
MEDS_HELD_OUT_0 = """
patient_id,code,timestamp,numerical_value
1500733,HEIGHT,,158.60131573580904
1500733,EYE_COLOR//BROWN,,
1500733,DOB,1986-07-20T00:00:00.000000,
1500733,TEMP,2010-06-03T14:54:38.000000,100.0
1500733,HR,2010-06-03T14:54:38.000000,91.4
1500733,ADMISSION//ORTHOPEDIC,2010-06-03T14:54:38.000000,
1500733,HR,2010-06-03T15:39:49.000000,84.4
1500733,TEMP,2010-06-03T15:39:49.000000,100.3
1500733,HR,2010-06-03T16:20:49.000000,90.1
1500733,TEMP,2010-06-03T16:20:49.000000,100.1
1500733,DISCHARGE,2010-06-03T16:44:26.000000,
"""
MEDS_TUNING_0 = """
patient_id,code,timestamp,numerical_value
754281,EYE_COLOR//BROWN,,
754281,HEIGHT,,166.22261567137025
754281,DOB,1988-12-19T00:00:00.000000,
754281,ADMISSION//PULMONARY,2010-01-03T06:27:59.000000,
754281,TEMP,2010-01-03T06:27:59.000000,99.8
754281,HR,2010-01-03T06:27:59.000000,142.0
754281,DISCHARGE,2010-01-03T08:22:13.000000,
"""

MEDS_OUTPUTS = {
    "train/0": MEDS_TRAIN_0,
    "train/1": MEDS_TRAIN_1,
    "held_out/0": MEDS_HELD_OUT_0,
    "tuning/0": MEDS_TUNING_0,
}

TRAIN_0_SCHEMA = {
    'patient_id': [239684, 1195293],
    'code': [
        [10, 8],
        [10, 7]],
    'numerical_value': [
        [1.577007, None],
        [0.068028, None]],
    'start_time': [
            "1980-12-28 00:00:00",
            "1978-06-20 00:00:00",],
    'timestamp': [
        ["1980-12-28 00:00:00",
        "2010-05-11 17:41:51",
        "2010-05-11 17:48:48",
        "2010-05-11 18:25:35",
        "2010-05-11 18:57:18",
        "2010-05-11 19:27:19",],
        ["1978-06-20 00:00:00",
        "2010-06-20 19:23:52",
        "2010-06-20 19:25:32",
        "2010-06-20 19:45:19",
        "2010-06-20 20:12:31",
        "2010-06-20 20:24:44",
        "2010-06-20 20:41:33",
        "2010-06-20 20:50:04",
        ]]
}

TRAIN_1_SCHEMA = {
    'patient_id': [68729, 814703],
    'code': [[9, 10], [9, 10]],
    'numerical_value': [[None, -0.543816], [None, -1.101219]],
    'start_time': ['1978-03-09 00:00:00', '1976-03-28 00:00:00'],
    'timestamp': [
        ['1978-03-09 00:00:00', '2010-05-26 02:30:56', '2010-05-26 04:51:52'],
        ['1976-03-28 00:00:00', '2010-02-05 05:55:39', '2010-02-05 07:02:30']
    ]
}

HELDOUT_0_SCHEMA = {
    'patient_id': [1500733],
    'code': [[10, 8]],
    'numerical_value': [[-0.799583, None]],
    'start_time': ['1986-07-20 00:00:00'],
    'timestamp': [
        ['1986-07-20 00:00:00', '2010-06-03 14:54:38', '2010-06-03 15:39:49', '2010-06-03 16:20:49', '2010-06-03 16:44:26']
    ]
}

TUNING_0_SCHEMA = {
    'patient_id': [754281],
    'code': [[8, 10]],
    'numerical_value': [[None, 0.286975]],
    'start_time': ['1988-12-19 00:00:00'],
    'timestamp': [
        ['2010-01-03 06:27:59', '2010-01-03 08:22:13', '1988-12-19 00:00:00']
    ]
}


def process_schema(schema):
    schema = pl.DataFrame(schema)
    schema = schema.with_columns([
        pl.col("start_time").str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S'),
        pl.col('timestamp').list.eval(pl.element().str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S'), parallel=True)
    ])
    return schema

def write_schema_df(schema_df, MEDS_cohort_dir, split, shard):
    output_dir = MEDS_cohort_dir / "tokenization" / "schemas" / split
    os.makedirs(output_dir, exist_ok=True)
    schema_df.write_parquet(output_dir / f"{shard}.parquet")


import numpy as np
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

# Define the data using numpy arrays for better handling of numerical data and NaN values
TRAIN_0_NRT_DATA = {
    'dim1/bounds': np.array([6, 14]),
    'dim1/lengths': np.array([6, 8]),
    'dim1/time_delta_days': [
        np.array([np.nan, 1.0726737e+04, 4.8263888e-03, 2.5543982e-02,
                  2.2025462e-02, 2.0844907e-02], dtype=np.float32),
        np.array([np.nan, 1.1688809e+04, 1.1574074e-03, 1.3738426e-02,
                  1.8888889e-02, 8.4837964e-03, 1.1678241e-02, 5.9143519e-03],
                 dtype=np.float32)
    ],
    'dim2/bounds': np.array([1, 4, 6, 8, 10, 11, 12, 15, 17, 19, 21, 23, 25, 26]),
    'dim2/code': [
        np.array([6.], dtype=np.float32),
        np.array([12., 2., 11.], dtype=np.float32),
        np.array([12., 11.], dtype=np.float32),
        np.array([12., 11.], dtype=np.float32),
        np.array([11., 12.], dtype=np.float32),
        np.array([5.], dtype=np.float32),
        np.array([6.], dtype=np.float32),
        np.array([12., 2., 11.], dtype=np.float32),
        np.array([12., 11.], dtype=np.float32),
        np.array([11., 12.], dtype=np.float32),
        np.array([11., 12.], dtype=np.float32),
        np.array([11., 12.], dtype=np.float32),
        np.array([12., 11.], dtype=np.float32),
        np.array([5.], dtype=np.float32)
    ],
    'dim2/lengths': np.array([1, 3, 2, 2, 2, 1, 1, 3, 2, 2, 2, 2, 2, 1]),
    'dim2/numerical_value': [
        np.array([np.nan], dtype=np.float32),
        np.array([-1.2713274, np.nan, -0.5697363], dtype=np.float32),
        np.array([-1.1678973, -0.43754688], dtype=np.float32),
        np.array([-1.3747574, 1.3218939e-03], dtype=np.float32),
        np.array([-0.04097871, -1.5299025], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([0.7972731, np.nan, -0.23133144], dtype=np.float32),
        np.array([0.7972731, 0.03833492], dtype=np.float32),
        np.array([0.33972675, 0.7455581], dtype=np.float32),
        np.array([-0.04626629, 0.69384307], dtype=np.float32),
        np.array([-0.30006993, 0.7972731], dtype=np.float32),
        np.array([1.0041331, -0.31064507], dtype=np.float32),
        np.array([np.nan], dtype=np.float32)
    ]
}

TRAIN_0_NRT_SCHEMA = {
    'dim1/time_delta_days': np.dtype('float32'),
    'dim2/code': np.dtype('float32'),
    'dim2/numerical_value': np.dtype('float32')
}
JNRT_TRAIN_0 = JointNestedRaggedTensorDict(TRAIN_0_NRT_DATA, TRAIN_0_NRT_SCHEMA, pre_raggedified=True)

import numpy as np
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

# Define the data using numpy arrays to handle numerical data and NaN values effectively
TRAIN_1_NRT_DATA = {
    'dim1/bounds': np.array([3, 6]),
    'dim1/lengths': np.array([3, 3]),
    'dim1/time_delta_days': [
        np.array([np.nan, 1.17661045e+04, 9.78703722e-02], dtype=np.float32),
        np.array([np.nan, 1.2367247e+04, 4.6423610e-02], dtype=np.float32)
    ],
    'dim2/bounds': np.array([1, 4, 5, 6, 9, 10]),
    'dim2/code': [
        np.array([6.], dtype=np.float32),
        np.array([11., 4., 12.], dtype=np.float32),
        np.array([5.], dtype=np.float32),
        np.array([6.], dtype=np.float32),
        np.array([12., 11., 3.], dtype=np.float32),
        np.array([5.], dtype=np.float32)
    ],
    'dim2/lengths': np.array([1, 3, 1, 1, 3, 1]),
    'dim2/numerical_value': [
        np.array([np.nan], dtype=np.float32),
        np.array([-1.4474739, np.nan, -0.34045714], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([0.8489881, 3.004665, np.nan], dtype=np.float32),
        np.array([np.nan], dtype=np.float32)
    ]
}

TRAIN_1_NRT_SCHEMA = {
    'dim1/time_delta_days': np.dtype('float32'),
    'dim2/code': np.dtype('float32'),
    'dim2/numerical_value': np.dtype('float32')
}

# Create the JointNestedRaggedTensorDict object
JNRT_TRAIN_1 = JointNestedRaggedTensorDict(TRAIN_1_NRT_DATA, TRAIN_1_NRT_SCHEMA, pre_raggedified=True)



def test_clmbr(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"
    describe_codes_config = {
        "raw_MEDS_cohort_dir": str(MEDS_cohort_dir.parent.resolve()),
        "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
        "max_seq_len": 512,
    }

    with initialize(
        version_base=None, config_path="../src/MEDS_torch/configs"
    ):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in describe_codes_config.items()]
        cfg = compose(config_name="pytorch_dataset", overrides=overrides)  # config.yaml

    # Create the directories
    MEDS_cohort_dir.mkdir(parents=True, exist_ok=True)

    # Store MEDS outputs
    for split, data in MEDS_OUTPUTS.items():
        file_path = MEDS_cohort_dir / f"{split}.parquet"
        file_path.parent.mkdir(exist_ok=True)
        df = pl.read_csv(StringIO(data))
        df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f")).write_parquet(
            file_path
        )

    # Check the files are not empty
    meds_files = list_subdir_files(Path(cfg.MEDS_cohort_dir), "parquet")
    assert (
        len(list_subdir_files(Path(cfg.MEDS_cohort_dir).parent, "parquet")) == 4
    ), "MEDS train split Data Files Should be 4!"
    for f in meds_files:
        assert pl.read_parquet(f).shape[0] > 0, "MEDS Data Tabular Dataframe Should not be Empty!"
    split_json = json.load(StringIO(SPLITS_JSON))
    splits_fp = MEDS_cohort_dir.parent / "splits.json"
    json.dump(split_json, splits_fp.open("w"))
    
    train_0_schema_df = process_schema(TRAIN_0_SCHEMA)
    train_1_schema_df = process_schema(TRAIN_1_SCHEMA)
    # heldout_0_schema_df = process_schema(HELDOUT_0_SCHEMA)
    # tuning_0_schema_df = process_schema(TUNING_0_SCHEMA)
    write_schema_df(train_0_schema_df, MEDS_cohort_dir, "train", "0")
    write_schema_df(train_1_schema_df, MEDS_cohort_dir, "train", "1")
    tensorize_dir = MEDS_cohort_dir / "tensorize"
    os.makedirs(tensorize_dir / "train", exist_ok=True)
    JNRT_TRAIN_0.save(tensorize_dir / "train/0.nrt")
    JNRT_TRAIN_1.save(tensorize_dir / "train/1.nrt")

    pyd = PytorchDataset(cfg, split="train")

    # Get an item:
    item = pyd[0]

    # Get a batch:
    batch = pyd.collate([pyd[i] for i in range(2)])
