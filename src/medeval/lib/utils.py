import json
from importlib.metadata import PackageNotFoundError, version

from typing import List, Dict


def is_installed(package_name, get_version=False):
    try:
        return version(package_name) if get_version else True
    except PackageNotFoundError:
        return None if get_version else False


def write_jsonl_data(path: str, data: List[Dict]):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
        f.close()


def write_json_data(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        f.close()
