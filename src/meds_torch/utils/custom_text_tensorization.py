#!/usr/bin/env python
"""Functions for tokenizing MEDS datasets.

Here, _tokenization_ refers specifically to the process of converting a longitudinal, irregularly sampled,
continuous time sequence into a temporal sequence at the level that will be consumed by deep-learning models.

All these functions take in _normalized_ data -- meaning data where there are _no longer_ any code modifiers,
as those have been normalized alongside codes into integer indices (in the output code column). The only
columns of concern here thus are `subject_id`, `time`, `code`, `numeric_value`.
"""

from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.utils import rwlock_wrap, shard_iterator
from MEDS_transforms.utils import hydra_loguru_init, write_lazyframe
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """TODO."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    output_dir = Path(cfg.stage_cfg.output_dir)
    # May need to fix this so it goes through JNRT
    shards_single_output, include_only_train = shard_iterator(cfg)
    meds_dir = Path(cfg.stage_cfg.meds_dir)

    if include_only_train:
        raise ValueError("Not supported for this stage.")

    for in_fp, out_fp in shards_single_output:
        nrt_path = out_fp.relative_to(output_dir)
        def read_fn(in_fp):
            return JointNestedRaggedTensor.load(in_fp)
        def compute_fn(jnrt):
            static_df = pl.read_parquet(schema_in_fp)
            for patient in patients_in_nrt(nrt_path):
                docs, times = load_patient_documents(patient, meds_dir)
                # get JNRT indexes of docs useing static_df like in get_task_indices_and_labels
                JNRT_indexes = get_JNRT_indexes(docs, static_df)
                # add docs to the JNRT
                jnrt.update(???)
                return jnrt
        def write_fn(jnrt, out_fp):
            jnrt.save(out_fp)

        rwlock_wrap(
            in_fp,
            schema_out_fp,
            read_fn,
            write_fn,
            compute_fn
            do_overwrite=cfg.do_overwrite,
        )

    logger.info(f"Done with {cfg.stage}")


if __name__ == "__main__":
    main()
