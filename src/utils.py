import yaml
from yaml import Loader
from typing import Dict

def load_config(cfg_path: str) -> Dict:
    with open(cfg_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader)

    return cfg