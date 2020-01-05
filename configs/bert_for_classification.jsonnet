/** You could basically use this config to train your own BERT classifier,
    with the following changes:
    1. change the `bert_model` variable to "bert-base-uncased" (or whichever you prefer)
    2. swap in your own DatasetReader. It should generate instances
       that look like {"tokens": TextField(...), "label": LabelField(...)}.
       You don't need to add the "[CLS]" or "[SEP]" tokens to your instances,
       that's handled automatically by the token indexer.
    3. replace train_data_path and validation_data_path with real paths
    4. any other parameters you want to change (e.g. dropout)
 */


# For a real model you'd want to use "bert-base-uncased" or similar.
local bert_model = "models/biobert_v1.1._pubmed/";

{
    "dataset_reader": {
        "lazy": false,
        "type": "new_ppi_dataset_reader",
        "tokenizer": {
            "type": "bert-basic"
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model
            }
        }
    },
    "train_data_path": "data/lin/train.jsonl",
    "validation_data_path": data/lin/dev.jsonl,
    "test_data_path": data/lin/train.jsonl,
    "evaluate_on_test": true,
    "model": {
        "type": "bert_for_classification",
        "bert_model": bert_model,
        "dropout": 0.0
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 5
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 4,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 0
    }
}