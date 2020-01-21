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
        "type": "ppi_dataset_reader",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained-special",
                "pretrained_model": bert_model,
                "do_lowercase": False,
                "max_pieces": 286,
                "use_starting_offsets": true,
                "additional_special_tokens": ["$","#"]
            }
        }
    },
    "train_data_path": "data/ppi_hu/lin/train.jsonl",
    "validation_data_path": "data/ppi_hu/lin/dev.jsonl",
    "test_data_path": "data/ppi_hu/lin/test.jsonl",
    "evaluate_on_test": true,
    "model": {
        "type": "bert_for_classification",
        "bert_model": bert_model,
        "dropout": 0.0
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 0.00002,
            "eps": 0.00000001
        },
        "validation_metric": "+average_F1",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 5,
        "should_log_learning_rate": true,
        "learning_rate_scheduler": {
          "type": "linear_schedule_with_warmup",
          "num_warmup_steps": 0,
          "num_training_steps": 10304
        },
        "cuda_device": 0
    }
}
