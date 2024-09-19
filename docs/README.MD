# Maintainer Guide

## Testing

To run tests, use the following command:

Run all the fast tests that are fast (and don't use gpu) with:

```bash
pytest -k "not slow"
```

Run all tests with:

```
pytest
```

______________________________________________________________________

## Local Documentation Development

This section explains how to edit documentation files in the `docs` directory.

First install docs code

```bash
pip install -e .[docs]
```

Run

```bash
mkdocs serve
```
