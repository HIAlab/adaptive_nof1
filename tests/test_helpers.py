from adaptive_nof1.helpers import *
import pytest

import pandas


def test_merge_with_postfix():
    dict_one = {"a": 1, "b": 2, "c": 3}
    dict_two = {"a": 3}

    expected = {"a#0": 1, "b#0": 2, "c#0": 3, "a#1": 3}

    assert merge_with_postfix([dict_one, dict_two]) == expected


def test_split_with_postfix():
    merged_dict = {"a#0": 1, "b#0": 2, "c#0": 3, "a#1": 3}

    dict_one = {"a": 1, "b": 2, "c": 3}
    dict_two = {"a": 3}

    assert split_with_postfix(merged_dict) == [dict_one, dict_two]


def test_index_from_subset():
    string_array = ["zero", "one", "two", "three", "four", "five"]
    number_array = list(range(100))

    string_subset = ["four", "two"]
    number_subset = [42, 1, 3]
    number_subset_reversed = [3, 1, 42]

    assert index_from_subset(string_array, string_subset) == [4, 2]

    assert index_from_subset(number_array, number_subset) == [42, 1, 3]
    assert index_from_subset(number_array, number_subset_reversed) == [3, 1, 42]

    assert index_from_subset([1, 2, 3], [1, 2, 3]) == [0, 1, 2]

    with pytest.raises(AssertionError):
        index_from_subset(string_array, number_subset)


def test_index_to_values():
    dimensions = [10, 10, 10]

    assert index_to_values(dimensions, 0) == [0, 0, 0]
    assert index_to_values(dimensions, 999) == [9, 9, 9]

    # Wraparound
    assert index_to_values(dimensions, 1000) == [0, 0, 0]

    dimensions = [2, 2, 4]
    index_value_pairs = [
        [0, [0, 0, 0]],
        [1, [0, 0, 1]],
        [2, [0, 0, 2]],
        [3, [0, 0, 3]],
        [4, [0, 1, 0]],
        [5, [0, 1, 1]],
        [6, [0, 1, 2]],
        [7, [0, 1, 3]],
        [8, [1, 0, 0]],
        [9, [1, 0, 1]],
    ]
    for pair in index_value_pairs:
        assert index_to_values(dimensions, pair[0]) == pair[1]


def test_values_to_index():
    dimensions = [2, 2, 4]
    index_value_pairs = [
        [0, [0, 0, 0]],
        [1, [0, 0, 1]],
        [2, [0, 0, 2]],
        [3, [0, 0, 3]],
        [4, [0, 1, 0]],
        [5, [0, 1, 1]],
        [6, [0, 1, 2]],
        [7, [0, 1, 3]],
        [8, [1, 0, 0]],
        [9, [1, 0, 1]],
    ]
    for pair in index_value_pairs:
        assert values_to_index(dimensions, pair[1]) == pair[0]


def test_index_to_value_counts():
    dimensions = [2, 2, 4]
    index_value_pairs = [
        [0, [3, 0, 0, 0]],
        [1, [2, 1, 0, 0]],
        [2, [2, 0, 1, 0]],
        [3, [2, 0, 0, 1]],
        [4, [2, 1, 0, 0]],
        [5, [1, 2, 0, 0]],
        [6, [1, 1, 1, 0]],
        [7, [1, 1, 0, 1]],
        [8, [2, 1, 0, 0]],
        [9, [1, 2, 0, 0]],
    ]
    for pair in index_value_pairs:
        assert index_to_value_counts(dimensions, pair[0]) == pair[1]


def test_index_to_actions():
    dimensions = [2, 2, 4]
    index = 6
    names = ["one", "two", "three"]
    assert index_to_actions(index, dimensions, names) == {
        "one": 0,
        "two": 1,
        "three": 2,
    }


def test_assert_all_equal():
    assert all_equal(["A", "A", "A"])
    assert not all_equal(["A", "A", "A", "B"])


def test_series_to_indexed_array():
    # Missing 0, and not in the correct order
    index = [3, 2, 1]
    values = [3, 2, 1]
    s = pandas.Series(values, index=index)
    assert series_to_indexed_array(s) == [0, 1, 2, 3]


def test_series_to_indexed_array_min_length():
    # Missing 0, and not in the correct order
    index = [3]
    values = [3]
    s = pandas.Series(values, index=index)
    assert series_to_indexed_array(s, min_length=5, fill_element=-1) == [
        -1,
        -1,
        -1,
        3,
        -1,
    ]
