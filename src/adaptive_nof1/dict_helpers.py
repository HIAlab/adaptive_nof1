from typing import Dict, List

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
