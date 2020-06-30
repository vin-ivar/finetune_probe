from argparse import ArgumentParser

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.util import import_module_and_submodules

from allennlp.data.dataset_readers import DatasetReader
from allennlp.models.model import Model
from allennlp.predictors import Predictor

import os
import torch
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
    args = parser.parse_args()

    import_module_and_submodules("model")
    import_module_and_submodules("loader")

    model_name = 'xlm-roberta-large' if args.model == 'xlmr' else 'bert-base-multilingual-cased'
    size = '1024' if args.model == 'xlmr' else '768'
    logger.info(f'Using model {model_name}')

    config = Params.from_file(args.config + '_train.jsonnet', ext_vars={'train_path': args.train,
                                                                        'val_path': args.val,
                                                                        'model_name': model_name,
                                                                        'model_size': size})

    model = train_model(config, args.save, force=True)

    val_partition = args.val
    reader = DatasetReader.from_params(config.duplicate().pop('dataset_reader'))

    predictor = Predictor(model, reader)
    uas, las = np.array([]), np.array([])

    # instances = reader.read(val_partition)
    # model.vocab.extend_from_instances(instances=instances)
    # if not args.finetune:
    #     model.extend_embedder_vocab()
    for i in reader._read(val_partition):
        _uas, _las, total = 0, 0, 0
        pred = predictor.predict_instance(i)
        gold_heads, gold_labels = i.fields['head_indices'].labels, i.fields['head_tags'].labels
        pred_heads, pred_labels = pred['predicted_heads'], pred['predicted_dependencies']

        assert len(gold_heads) == len(gold_labels) == len(pred_heads) == len(pred_labels)
        for (gh, gl, ph, pl) in zip(gold_heads, gold_labels, pred_heads, pred_labels):
            if ph == gh:
                _uas += 1.

            if ph == gh and pl == gl:
                _las += 1.

            total += 1
        uas = np.append(uas, _uas / total)
        las = np.append(las, _las / total)

    print("! val macro-avg uas: {}, las: {}".format(uas.mean(), las.mean()))

main()
