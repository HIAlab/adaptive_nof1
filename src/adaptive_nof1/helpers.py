from typing import Dict, List
import itertools
import numpy

seperator = "#"


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
    assert set(subset) <= set(array), "Given Subset is not a subset of array"
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

# Counts the number of occurences of each number an returns a list.
# List[a] == number of occurences of a in values
def index_to_value_counts(dimensions, index):
    values = index_to_values(dimensions, index)
    return list(numpy.bincount(values, minlength=max(dimensions)))


def index_to_actions(index, dimensions, names):
    values = index_to_values(dimensions, index)
    return {name: action for name, action in zip(names, values)}


def series_to_indexed_array(series, fill_element=0, min_length=0):
    array = [fill_element] * min_length
    for index, value in series.items():
        if index + 1 > len(array):
            array.extend([fill_element] * (index + 1 - len(array)))
        array[index] = value

    return array


def all_equal(x):
    return x.count(x[0]) == len(x)


def flatten(x):
    return [item for row in x for item in row]


def flatten_dictionary(dict):
    new_dict = {}
    for key, value in dict.items():
        if isinstance(value, (list, numpy.ndarray)):
            # If the value is a list or NumPy array, iterate through its elements
            for i, elem in enumerate(value):
                new_key = f"{key}_{i}"
                new_dict[new_key] = elem
        else:
            # For non-list or non-array values, just copy them to the new dictionary
            new_dict[key] = value

    return new_dict


def array_almost_equal(one, two, epsilon=0.01):
    for a, b in zip(one, two):
        if abs(a - b) > epsilon:
            return False
    return True


def generate_configuration_cross_product(configuration_specification):
    configuration_dimensions = [
        list(range(len(item))) for item in configuration_specification.values()
    ]
    configuration_indices = list(itertools.product(*configuration_dimensions))
    parameter_names = list(configuration_specification.keys())
    parameter_value_array = list(configuration_specification.values())
    configurations = [
        {
            parameter_names[index]: parameter_value_array[index][value]
            for index, value in enumerate(configuration_index)
        }
        for configuration_index in configuration_indices
    ]
    return configurations


def contains_keys(dict, keys):
    return all(key in dict for key in keys)
