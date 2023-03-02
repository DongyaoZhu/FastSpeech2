import numpy as np
from pprint import pformat
import yaml


class Config:
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


def load_hparams(path):
    with open(path) as stream:
        hps = yaml.load(stream, yaml.Loader)
    hps = Config(hps)
    return hps


hps = load_hparams('fastspeech2.yaml')
# hps = load_hparams('recipes/LJSpeech/tts/fastspeech2/fastspeech2.yaml')
