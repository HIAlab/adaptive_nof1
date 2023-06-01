from adaptive_nof1.dict_helpers import merge_with_postfix, split_with_postfix


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
