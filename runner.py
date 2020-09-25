from argparse import ArgumentParser

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.util import import_module_and_submodules

from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.models.model import Model
from allennlp.training.trainer import Trainer

from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.common import util as common_util
from allennlp.training.metric_tracker import MetricTracker


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
    parser.add_argument('--epochs', action='store', type=int)
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
                                                     'lca': args.lca,
                                                     'freeze': args.freeze})

    training_util.create_serialization_dir(config, args.save, recover=False, force=True)
    common_util.prepare_environment(config)
    common_util.prepare_global_logging(args.save, True)

    reader = DatasetReader.from_params(config.pop('dataset_reader'))
    train_data = reader.read(config.pop('train_data_path'))
    val_data = reader.read(config.pop('validation_data_path'))

    vocab = Vocabulary.from_instances(train_data + val_data)
    model = Model.from_params(config.pop('model'), vocab=vocab)

    train_data.index_with(vocab)
    val_data.index_with(vocab)

    train_loader = DataLoader.from_params(config.get('data_loader'), dataset=train_data)
    val_loader = DataLoader.from_params(config.get('data_loader'), dataset=val_data)

    trainer = Trainer.from_params(config.pop('trainer'), model=model, serialization_dir='/tmp',
                                  data_loader=train_loader, validation_data_loader=val_loader)
    optimizer = trainer.optimizer
    cuda_device = trainer.cuda_device

    if cuda_device != -1:
        model.cuda(cuda_device)

    for epoch in range(args.epochs):
        metrics = {}
        batches_this_epoch = 0
        train_loss = 0.0

        model.train()
        for batch in tqdm.tqdm(train_loader):
            batch = nn_util.move_to_device(batch, cuda_device)
            optimizer.zero_grad()

            output_dict = model(**batch)
            loss = output_dict['loss']
            loss.backward()
            optimizer.step()

            model.log_lca()
            train_loss += loss.item()
            batches_this_epoch += 1

            train_metrics = training_util.get_metrics(model, train_loss, train_loss, batches_this_epoch,
                                                      reset=False, cuda_device=[cuda_device])
        # eval
        batches_this_epoch = 0
        val_loss = 0.0

        model.eval()
        for batch in tqdm.tqdm(val_loader):
            batch = nn_util.move_to_device(batch, cuda_device)
            with torch.no_grad():
                output_dict = model(**batch)
                loss = output_dict['loss']
                val_loss += loss.item()
                batches_this_epoch += 1

            val_metrics = training_util.get_metrics(model, val_loss, val_loss, batches_this_epoch,
                                                    reset=False, cuda_device=[cuda_device])

        trainer._metric_tracker.add_metric(val_metrics[trainer._validation_metric])
        metrics['epoch'] = epoch
        for key, value in train_metrics.items():
            metrics[f"training_{key}"] = value
        for key, value in val_metrics.items():
            metrics[f"validation_{key}"] = value

        if trainer._metric_tracker.is_best_so_far():
            metrics["best_epoch"] = epoch
            for key, value in val_metrics.items():
                metrics[f"best_validation_{key}"] = value
            trainer._metric_tracker.best_epoch_metrics = val_metrics

        common_util.dump_metrics(os.path.join(args.save, 'metrics.json'), metrics=metrics, log=True)


main()
