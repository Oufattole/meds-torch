# Input Encoders

This folder contains various input encoder implementations for processing and embedding different types of data in machine learning models.

## Files

### 1. text_encoder.py

This file contains two main classes:

- `TextCodeEncoder`: Encodes triplets of (time, code, value) data.
- `TextObservationEncoder`: Encodes observation tokens.

Both classes use an `AutoEmbedder` which is based on a pre-trained model from the Hugging Face Transformers library.

### 2. triplet_encoder.py

This file implements a `TripletEncoder` class that encodes triplets of (time, code, numeric_value) data. It uses separate embedders for each component of the triplet:

- `CVE (Continuous Value Encoder)`: For encoding times and numeric values.
- `nn.Embedding`: For encoding codes.

### 3. triplet_prompt_encoder.py

This file contains a `TripletPromptEncoder` class, which is similar to the `TripletEncoder` but specifically designed for prompt-based tasks. It encodes (code, numeric_value) pairs without the time component.
