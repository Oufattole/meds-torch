# Methods for Tokenizing MEDS Data

Let's explore different tokenization strategies using a unified example: a patient with two lab observations - a potassium lab and a creatinine lab - taken one day apart.

### Everything Is a CODE (EIC)

In this approach, we convert all information into discrete tokens, including lab values and time gaps.

Example:

1. Potassium lab
2. Quantile 9/10
3. 1-day time gap
4. Creatinine lab
5. Quantile 2/10

Alternatively, we could combine `codes` with their corresponding `numeric_value` elements:

1. Potassium lab quantile 9/10
2. 1-day time gap
3. Creatinine lab quantile 2/10

### Everything Is Text

This strategy converts all data into text format, which can then be processed by a language model.

Example text representation:
"code potassium lab value 5.2. time One day code creatinine lab value 0.7."

We can apply this approach at different levels:

1. Code text: "potassium lab", "creatinine lab"
2. Observation text: "potassium lab 5.2", "time One day code creatinine lab value 0.7"
3. All text: The entire patient history as a narrative: "code potassium lab value 5.2. time One day code creatinine lab value 0.7."

Process:

1. Convert data to text (at code, observation, or patient history level)
2. Use a language model to tokenize the text
3. Generate embeddings for downstream tasks

### Triplet

This method represents each observation as a combination of time, code, and value.

Example:

1. (Day 0, Potassium, 5.2)
2. (Day 1, Creatinine, 0.7)

Each triplet is encoded by embedding its components separately and then summing the resulting vectors.

<!-- ### Prompt Expanded Triplet

This approach structures the sequence with explicit markers for each component.

Example:
"[TS] [TS = Day 0] [CODE: Potassium] [VAL] [VAL = 5.2] [TS] [TS = Day 1] [CODE: Creatinine] [VAL] [VAL = 0.7]"

This method could allow an autoregressive generative model to better express conditional predictions for values or timestamps. -->

### Event Stream

This method groups all observations occurring at the same time.

Example:
\[(Day 0, \[(Potassium, 5.2)\]),
(Day 1, \[(Creatinine, 0.7)\])\]
