from argparse import ArgumentParser
from _jsonnet import evaluate_file, evaluate_snippet

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
    parser.add_argument('--save', action='store', default='experiments/models')
    # kill args
    parser.add_argument('--kill', action='store', type=str)
    # LCA args
    parser.add_argument('--lca', action='store', type=str)
    parser.add_argument('--lca_mode', action='store', type=str)
    # other
    parser.add_argument('--epochs', action='store', type=str)
    parser.add_argument('--seed', action='store', type=str, default='42')
    args = parser.parse_args()

    import_module_and_submodules("components.loader")
    import_module_and_submodules("components.model")
    import_module_and_submodules("components.trainer")

    model_name = 'xlm-roberta-large' if args.model == 'xlmr' else 'bert-base-multilingual-cased'
    size = '1024' if args.model == 'xlmr' else '768'
    logger.info(f'Using model {model_name}')

    var_dict = {'train_path': args.train, 'val_path': args.val, 'epochs': args.epochs, 'seed': args.seed,
                'model_name': model_name, 'model_size': size,
                'kill': args.kill, 'lca': args.lca, 'lca_mode': args.lca_mode}

    config = Params.from_file(args.config, ext_vars=var_dict)
    model = train_model(config, args.save, force=True)


main()
