# Definitions for meds-torch Components

In addition to the terms defined in the [official MEDS Schema](https://github.com/Medical-Event-Data-Standard/meds) and [MEDS_transforms](https://meds-transforms.readthedocs.io/en/latest/terminology/), we define the following terms for use in meds-torch:

#### Backbone

The sequence model used to process and analyze medical event data.

#### Input Encoder

The model or algorithm that converts raw data (stored on disk and output by the final `tensorize` stage in the `MEDS_transform` repository) into a sequence of tokens that can be fed to the backbone.

#### Triplet

A representation of any observation as a tuple of (`time`, `code`, `numeric_value`). Triplets are encoded as tokens by embedding each of the three elements separately into equal-dimensional vectors and summing these vectors together.
