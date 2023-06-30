from typing import Dict, List

seperator = "#"

# TODO: Test (all)


def merge_with_postfix(dicts: List[Dict]) -> Dict:
    merged = {}
    for index, dict in enumerate(dicts):
        merged.update(
            {f"{key}{seperator}{index}": value for key, value in dict.items()}
        )
    return merged


def split_with_postfix(dict: Dict) -> List[Dict]:
    reconstruction = []
    for key, value in dict.items():
        name, index_string, *_ = key.split(seperator)
        index = int(index_string)
        while len(reconstruction) < index + 1:
            reconstruction += [{}]
        reconstruction[index].update({name: value})
    return reconstruction


def index_from_subset(array, subset):
    assert set(subset) < set(array), "Given Subset is not a subset of array"
    return [array.index(string) for string in subset]


def index_to_values(dimensions, index):
    values = []
    for dimension in reversed(dimensions):
        values.append(index % dimension)
        index //= dimension
    values.reverse()

    return values


def values_to_index(dimensions, values):
    index = 0
    for i in range(len(dimensions)):
        index += values[i]
        if i < len(dimensions) - 1:
            index *= dimensions[i + 1]
    return index


def index_to_actions(index, dimensions, names):
    values = index_to_values(dimensions, index)
    return {name: action for name, action in zip(names, values)}
