# Pipelines

## Step01 - Corpus

Cleans and turns text collections into form that is ready to be tokenized
Inputs:
 * Text files
 * EstNLTK Text objects or collections
 * Standard text corpuses

Results:
 * Text file? that is ready to be tokenized

## Step02 - BERT pre-training

Uses the texts processed in the previous step to:
 * Create a vocabulary (if not provided)
   * Will end up in a file like `data/test_model_1234125352/vocab.txt`
 * Tokenize the texts (if not provided | save if required)
   * In data preparation step, while initiating tokenization, following files
     are attempted to load from `data/test_model_1234125352/`
      * `tokenizer.json`
      * `added_tokens.json`
      * `special_tokens_map.json`
      * `tokenizer_config.json`
      * `vocab.txt`
 * pre-train BERT

While pre-training, it also displays diagnostics (MLM and NSP scores).

### BERT pre-training result files

Model pre-training saves checkpoints and final results (including config):

Example of checkpoints/results saved:

* `data/test_model_1234125352{/checkpoint-500}`
  * `config.json` - model configuration aka BertConfig
  * `pytorch_model.bin` - model weights, used for initializing BertForPreTraining
  * `tokenizer_config.json` - tokenizer config file
  * `special_tokens_map.json` - special tokens file

At this point, if your task is similar to the task the model of the checkpoint was trained on, 
you can already use BertForPreTraining for predictions without further training.

File `config.json` looks something like following:

```
{
  "architectures": [
    "BertForPreTrainingMod"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 480,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 1024,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.8.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 4000
}
```


## Step03 - BERT fine-tuning

Fine-tunes a BERT model

Inputs:
 * pre-trained model
 * tagged text files OR tagged EstNLTK collections

