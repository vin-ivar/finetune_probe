from argparse import ArgumentParser

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.util import import_module_and_submodules

from allennlp.data.dataset_readers import DatasetReader
from allennlp.models.model import Model
from allennlp.predictors import Predictor

import os
import torch
import pickle
import numpy as np

import logging

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser()
    parser.add_argument('--train', action='store')
    parser.add_argument('--val', action='store')
    parser.add_argument('--model', action='store', default='bert')
    parser.add_argument('--config', action='store', default='configs/cpu')
    parser.add_argument('--save', action='store', default='experiments/models/default')
    parser.add_argument('--param_save', action='store')
    parser.add_argument('--param_freeze', action='store')
    args = parser.parse_args()

    import_module_and_submodules("model")
    import_module_and_submodules("loader")

    model_name = 'xlm-roberta-large' if args.model == 'xlmr' else 'bert-base-multilingual-cased'
    size = '1024' if args.model == 'xlmr' else '768'
    logger.info(f'Using model {model_name}')

    config = Params.from_file(args.config, ext_vars={'train_path': args.train,
                                                     'val_path': args.val,
                                                     'model_name': model_name,
                                                     'model_size': size,
                                                     'param_freeze': args.param_freeze})
    model = train_model(config, args.save, force=True)
    if args.param_save:
        with open(args.param_save, 'wb') as f:
            pickle.dump(model._params_to_log, f)


main()
