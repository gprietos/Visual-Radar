"""Config loading.

Configuration lives in a YAML file (see configs/default.yaml), loaded as a
plain nested dict rather than typed dataclasses. Each building block still
takes its own explicit constructor args; callers unpack the relevant YAML
section into those args instead of passing the whole config around.
"""


import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file into a plain nested dict. Wraps `yaml.safe_load`."""
    with open(path) as f:
        return yaml.safe_load(f)
