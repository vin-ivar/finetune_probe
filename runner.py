from argparse import ArgumentParser

from allennlp.commands.train import train_model
from allennlp.common import Params, Tqdm
from allennlp.common.util import import_module_and_submodules

from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.models.model import Model
from allennlp.training.trainer import Trainer

from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.common import util as common_util
from allennlp.common.logging import prepare_global_logging

from torch.nn.utils import clip_grad_norm_

import tensorboard
import torch
import tqdm
import json

import os

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

def main():
    parser = ArgumentParser()
    parser.add_argument('--train', action='store')
    parser.add_argument('--val', action='store')
    parser.add_argument('--model', action='store', default='bert')
    parser.add_argument('--config', action='store', default='configs/cpu')
    parser.add_argument('--save', action='store', default='experiments/models/default/test/deep')
    parser.add_argument('--freeze', action='store', type=str)
    parser.add_argument('--lca', action='store', type=str)
    parser.add_argument('--epochs', action='store', type=str)
    args = parser.parse_args()

    import_module_and_submodules("components.loader")
    import_module_and_submodules("components.model")
    import_module_and_submodules("components.trainer")

    model_name = 'xlm-roberta-large' if args.model == 'xlmr' else 'bert-base-multilingual-cased'
    size = '1024' if args.model == 'xlmr' else '768'
    logger.info(f'Using model {model_name}')

    config = Params.from_file(args.config, ext_vars={'train_path': args.train, 'val_path': args.val,
                                                     'model_name': model_name, 'model_size': size,
                                                     'lca': args.lca, 'freeze': args.freeze, 'epochs': args.epochs})

    model = train_model(config, args.save, force=True)


main()
