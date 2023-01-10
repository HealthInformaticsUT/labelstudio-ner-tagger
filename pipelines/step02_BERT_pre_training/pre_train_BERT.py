import os
from typing import Optional, List, Tuple, Callable

from os import listdir
from os.path import isfile, join

from datasets import load_dataset
import torch

from transformers import BertConfig, BertTokenizerFast, DataCollatorForLanguageModeling, \
    TrainingArguments, TrainerCallback, PreTrainedModel

from pipelines.step02_BERT_pre_training.pre_training.BertForPreTrainingMod import BertForPreTrainingMod
from pipelines.step02_BERT_pre_training.pre_training.Helpers import training_args_deprecation_fix
from pipelines.step02_BERT_pre_training.pre_training.PreTrainer import PreTrainer
from pipelines.step02_BERT_pre_training.tokenizing.text_dataset_for_NSP import create_dataset_for_NSP
from pipelines.step02_BERT_pre_training.tokenizing.vocabulary_creator import create_vocabulary


def pre_train_BERT(model_path: str, input_files, training_args: dict = None, vocab_size: int = 3000,
                   bert_config: BertConfig = None, mlm_probability: float = 0.15,
                   nsp_probability: float = 0.5,
                   model_init: Callable[[], PreTrainedModel] = None,
                   callbacks: Optional[List[TrainerCallback]] = None,
                   optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                   **kwargs):
    """
    :param model_path: (string) path to the model directory, if the directory does not exist, then it is created
    :param input_files: (string or [string]) path(s) to the input .tsv files
    :param training_args: (dict, default=None), Arguments used in training. If None, the default training
     args will be used (except do_train will be True and output_dir will be the model path). (See: https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments)
    :param vocab_size: (int) The number of tokens to add into the vocabulary or is in the provided vocabulary (including the    special tokens)
    :param bert_config: (BertConfig object, default=None) bert config (See https://huggingface.co/transformers/model_doc/bert.html#bertconfig)
    :param mlm_probability: (float, default=0.15), the probability that a token is masked.
    :param nsp_probability: (float, default=0.5), the probability that the next sentence is consecutive.
    :param model_init: A function that instantiates the model to be used. If provided, each call to train() will start from a new instance of the model as given by this function. (see Trainer doc)
    :param callbacks: A list of callbacks to customize the training loop. (see https://huggingface.co/transformers/main_classes/callback.html)
    :param optimizers: A tuple containing the optimizer and the scheduler to use. Will default to an instance of AdamW on your model and a scheduler given by get_linear_schedule_with_warmup() controlled by args.
    :param kwargs:
     **BertConfig** **kwargs: Only used if bert_config is None and there is no config file in model_path (See: https://huggingface.co/transformers/model_doc/bert.html#bertconfig)
     **BertTokenizerFast** **kwargs:  (See: https://huggingface.co/transformers/model_doc/bert.html#berttokenizerfast)
     **Tokenizer \_\_call__** **kwargs: (See: https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__)
    :return: pre-trained BERT model
    """

    if 'special_tokens' in kwargs:
        kwargs['special_tokens'] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    # if model dir does not exist, then create it
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    model_dir_files = [f for f in listdir(model_path) if isfile(join(model_path, f))]

    # if config wasn't provided but is in model dir, then use it.
    if bert_config is None and "config.json" in model_dir_files:
        bert_config = BertConfig.from_pretrained(model_path)
        os.system("echo Config was loaded from a file")

    # if bert_config is still None, then create a new one
    elif bert_config is None:
        bert_config = BertConfig(vocab_size=vocab_size, **kwargs)
        os.system("echo New config was created")

    # if there is a vocab_size in bert config, then replace it in the config
    if vocab_size > 0 and "vocab_size" in bert_config.to_dict().keys():
        bert_config.vocab_size = vocab_size

    # saving the config
    bert_config.save_pretrained(model_path)

    # if vocabulary is not provided, then create a new vocabulary
    if "vocab.txt" not in model_dir_files:
        create_vocabulary(model_path, input_files, size=vocab_size, **kwargs)
        os.system("echo New vocabulary was created")
    else:
        os.system("echo Vocabulary loaded from file")

    # DATA PREP
    os.system("echo Beginning tokenization")
    tokenizer = BertTokenizerFast.from_pretrained(model_path, **kwargs)
    dataset = load_dataset("csv", data_files={'train': input_files})['train']
    dataset_enc = create_dataset_for_NSP(dataset, tokenizer, nsp_probability=nsp_probability, **kwargs)

    # TRAINING
    os.system("echo Dataset tokenized, beginning training.")
    model = BertForPreTrainingMod(bert_config)

    # if training_args is not provided, then create a basic one
    if training_args is None:
        training_args = {"output_dir": model_path, "do_train": True}

    # initializing the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )

    # initializing the trainer
    trainer = PreTrainer(
        model=model,
        args=TrainingArguments(**training_args_deprecation_fix(training_args)),
        data_collator=data_collator,
        train_dataset=dataset_enc,
        tokenizer=tokenizer,
        model_init=model_init,
        callbacks=callbacks,
        optimizers=optimizers,
    )
    # training and saving the final result
    trainer.train()
    trainer.save_model(model_path)
