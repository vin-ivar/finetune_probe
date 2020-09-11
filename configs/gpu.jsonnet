{
    "dataset_reader": {
        "type":  "wordpiece_ud",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": std.extVar("model_name"),
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": std.extVar("model_name"),
            }
        }
    },
    "train_data_path": std.extVar("train_path"),
    "validation_data_path": std.extVar("val_path"),
    "model": {
      "type": "wordpiece_parser",
      "freezer": std.extVar("param_freeze"),
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "pretrained_transformer",
            "model_name": std.extVar("model_name"),
          },
        },
      },
      "encoder": {
        "type": "feedforward",
        "input_dim": std.parseInt(std.extVar("model_size")),
        "num_layers": 1,
        "hidden_dims": 400,
        "activations": "relu",
      },
      "use_mst_decoding_for_validation": false,
      "arc_representation_dim": 500,
      "tag_representation_dim": 100,
      "dropout": 0.3,
      "input_dropout": 0.3,
      "initializer": {
        "regexes": [
          [".*projection.*weight", {"type": "xavier_uniform"}],
          [".*projection.*bias", {"type": "zero"}],
          [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
          [".*tag_bilinear.*bias", {"type": "zero"}],
          [".*weight_ih.*", {"type": "xavier_uniform"}],
          [".*weight_hh.*", {"type": "orthogonal"}],
          [".*bias_ih.*", {"type": "zero"}],
          [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
        ],
      },
    },
    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "sorting_keys": ["words"],
        "batch_size" : 4
      },
    },
    "trainer": {
      "num_epochs": 50,
      "grad_norm": 5.0,
      "patience": 50,
      "cuda_device": 0,
      "validation_metric": "+LAS",
      "optimizer": {
        "type": "dense_sparse_adam",
     	"lr": 3e-5,
        "betas": [0.9, 0.9]
      }
    }
  }