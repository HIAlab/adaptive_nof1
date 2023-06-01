
from adaptive_nof1.models.model import merge_with_postfix, split_with_postfix


def test_merge_with_postfix():
    dict_one = {"a": 1, "b": 2, "c": 3}
    dict_two = {"a": 3}

    expected = {"a_0": 1, "b_0": 2, "c_0": 3, "a_1": 3}

    assert merge_with_postfix([dict_one, dict_two]) == expected

def test_split_with_postfix():
    merged_dict = {"a_0": 1, "b_0": 2, "c_0": 3, "a_1": 3}

    dict_one = {"a": 1, "b": 2, "c": 3}
    dict_two = {"a": 3}

    assert split_with_postfix(merged_dict) == [dict_one, dict_two]
