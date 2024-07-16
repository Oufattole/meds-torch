import datetime

import numpy as np
import polars as pl
import pytest

from meds_torch import EIToken


@pytest.fixture
def sample_data():
    full_data = {
        "patient_id": [
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
        ],
        "timestamp": [
            datetime.datetime(2012, 1, 1, 12, 0),
            datetime.datetime(2012, 10, 28, 12, 0),
            datetime.datetime(2013, 8, 24, 12, 0),
            datetime.datetime(2013, 8, 25, 12, 0),
            datetime.datetime(2013, 9, 23, 12, 0),
            datetime.datetime(2013, 9, 24, 12, 0),
            datetime.datetime(2013, 10, 23, 12, 0),
            datetime.datetime(2013, 10, 24, 12, 0),
            datetime.datetime(2014, 8, 19, 12, 0),
            datetime.datetime(2014, 8, 19, 12, 56),
            datetime.datetime(2014, 8, 20, 2, 56),
            datetime.datetime(2014, 8, 20, 3, 52),
            datetime.datetime(2014, 8, 21, 3, 52),
            datetime.datetime(2014, 8, 22, 3, 52),
            datetime.datetime(2014, 8, 23, 3, 52),
            datetime.datetime(2014, 8, 23, 17, 52),
            datetime.datetime(2014, 8, 24, 7, 52),
            datetime.datetime(2014, 8, 24, 21, 52),
            datetime.datetime(2014, 8, 24, 22, 48),
            datetime.datetime(2015, 6, 20, 22, 48),
            datetime.datetime(2016, 4, 16, 22, 48),
            datetime.datetime(2016, 4, 16, 23, 44),
        ],
        "code": [
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "C",
        ],
        "numerical_value": [
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
            18.0,
            20.0,
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
            18.0,
            20.0,
            22.0,
            24.0,
            5.0,
        ],
    }

    empty_data = {"patient_id": [], "timestamp": [], "code": [], "numerical_value": []}

    timestamp_missing = {
        "patient_id": [
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
        ],
        "timestamp": [
            datetime.datetime(2012, 1, 1, 12, 0),
            datetime.datetime(2012, 10, 28, 12, 0),
            datetime.datetime(2013, 8, 24, 12, 0),
            datetime.datetime(2013, 8, 25, 12, 0),
            datetime.datetime(2013, 9, 23, 12, 0),
            datetime.datetime(2013, 9, 24, 12, 0),
            datetime.datetime(2013, 10, 23, 12, 0),
            datetime.datetime(2013, 10, 24, 12, 0),
            datetime.datetime(2014, 8, 19, 12, 0),
            datetime.datetime(2014, 8, 19, 12, 56),
            datetime.datetime(2014, 8, 20, 2, 56),
            datetime.datetime(2014, 8, 20, 3, 52),
            datetime.datetime(2014, 8, 21, 3, 52),
            datetime.datetime(2014, 8, 22, 3, 52),
            datetime.datetime(2014, 8, 23, 3, 52),
            datetime.datetime(2014, 8, 23, 17, 52),
            datetime.datetime(2014, 8, 24, 7, 52),
            datetime.datetime(2014, 8, 24, 21, 52),
            datetime.datetime(2014, 8, 24, 22, 48),
            datetime.datetime(2015, 6, 20, 22, 48),
            datetime.datetime(2016, 4, 16, 22, 48),
            None,
        ],
        "code": [
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "C",
        ],
        "numerical_value": [
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
            18.0,
            20.0,
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
            18.0,
            20.0,
            22.0,
            24.0,
            5.0,
        ],
    }

    value_missing = {
        "patient_id": [
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
        ],
        "timestamp": [
            datetime.datetime(2012, 1, 1, 12, 0),
            datetime.datetime(2012, 10, 28, 12, 0),
            datetime.datetime(2013, 8, 24, 12, 0),
            datetime.datetime(2013, 8, 25, 12, 0),
            datetime.datetime(2013, 9, 23, 12, 0),
            datetime.datetime(2013, 9, 24, 12, 0),
            datetime.datetime(2013, 10, 23, 12, 0),
            datetime.datetime(2013, 10, 24, 12, 0),
            datetime.datetime(2014, 8, 19, 12, 0),
            datetime.datetime(2014, 8, 19, 12, 56),
            datetime.datetime(2014, 8, 20, 2, 56),
            datetime.datetime(2014, 8, 20, 3, 52),
            datetime.datetime(2014, 8, 21, 3, 52),
            datetime.datetime(2014, 8, 22, 3, 52),
            datetime.datetime(2014, 8, 23, 3, 52),
            datetime.datetime(2014, 8, 23, 17, 52),
            datetime.datetime(2014, 8, 24, 7, 52),
            datetime.datetime(2014, 8, 24, 21, 52),
            datetime.datetime(2014, 8, 24, 22, 48),
            datetime.datetime(2015, 6, 20, 22, 48),
            datetime.datetime(2016, 4, 16, 22, 48),
            datetime.datetime(2016, 4, 16, 23, 44),
        ],
        "code": [
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "C",
        ],
        "numerical_value": [
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
            18.0,
            20.0,
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
            18.0,
            20.0,
            22.0,
            24.0,
            None,
        ],
    }

    return {
        "full_data": pl.DataFrame(full_data),
        "missing_value": pl.DataFrame(value_missing),
        "missing_timestamp": pl.DataFrame(timestamp_missing),
        "empty_data": pl.DataFrame(empty_data),
    }


def test_full_data():
    data = {
        "patient_id": ["1", "1", "1", "1", "1", "1", "2", "2", "2", "2"],
        "code": ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A"],
        "numerical_value": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
    }
    df = pl.DataFrame(data)
    result_df = EIToken.calculate_quantiles(df)
    expected_data = {"code": ["A", "B"]}
    for i in range(1, 11):
        expected_data[str(i)] = [1 + 0.1 * i, 3 + 0.1 * i]
    expected_df = pl.DataFrame(expected_data)
    assert result_df.equals(expected_df, null_equal=True), "Full data test failed"


def test_value_missing():
    data = {"code": ["A", "A", "B", "B"], "numerical_value": [1.0, np.nan, 3.0, 4.0]}
    df = pl.DataFrame(data)
    result_df = EIToken.calculate_quantiles(df)
    expected_data = {"code": ["A", "B"]}
    expected_data["1"] = [1.0, 3.1]
    for i in range(2, 11):
        expected_data[str(i)] = [1.0, 3 + 0.1 * i]
    expected_df = pl.DataFrame(expected_data)
    assert result_df.equals(expected_df, null_equal=True), "One missing value test failed"


def test_timestamp_missing():
    data = {"code": ["A", "A", "B", "B"], "numerical_value": [np.nan, np.nan, np.nan, np.nan]}
    df = pl.DataFrame(data)
    result_df = EIToken.calculate_quantiles(df)
    expected_columns = ["code"] + [str(i) for i in range(1, 11)]
    expected_df = pl.DataFrame({col: [] for col in expected_columns}).with_columns(
        pl.Series("code", ["A", "B"])
    )
    assert result_df.equals(expected_df, null_equal=True), "All values missing test failed"


def test_empty_data():
    df_empty = pl.DataFrame({"code": [], "numerical_value": []})
    result_df = EIToken.calculate_quantiles(df_empty)
    expected_columns = ["code"] + [str(i) for i in range(1, 11)]
    expected_df = pl.DataFrame({col: [] for col in expected_columns})
    assert result_df.equals(expected_df, null_equal=True), "Empty DataFrame test failed"
