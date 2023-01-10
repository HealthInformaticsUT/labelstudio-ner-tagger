import csv
from typing import Optional, Tuple, List, Union

import datasets
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import TrainerCallback, AutoTokenizer, BertConfig, BertForTokenClassification, Trainer, \
    TrainingArguments

from pipelines.step02_BERT_pre_training.pre_training.Helpers import training_args_deprecation_fix
from pipelines.step03_BERT_fine_tuning.dataloaders import Tokens


def finetune_BERT(model_path: str,
                  data_loader: Union[Tokens.Tsv, Tokens.EstNLTKCol],
                  label_all_subword_units: bool = True,
                  save_model: bool = True,
                  map_args: Optional[dict] = None,
                  tokenizer_args: Optional[dict] = None,
                  tokenization_args: Optional[dict] = None,
                  training_args: dict = None,
                  compute_metrics=None,
                  callbacks: Optional[List[TrainerCallback]] = None,
                  optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)):
    """
    Fine-tunes a bert model on a sequence classification task
    :param model_path: (str) Path to the pretrained BERT model
    :param data_loader: (Union[Sequences.Tsv, Sequences.EstNLTKCol]) A class that is used to read the input data
    :param label_all_subword_units: (bool = True) to label all subword units or only the first
    :param save_model: (bool, default=True) To save a model or not
    :param map_args: (Optional[dict] = None) See: https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=map#datasets.Dataset.map
    :param tokenizer_args: (Optional[dict] = None) See: https://huggingface.co/transformers/model_doc/bert.html#berttokenizerfast
    :param tokenization_args: (Optional[dict] = None) See: https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
    :param training_args: (Optional[dict] = None), Arguments used in training. If None, the default training
         args will be used (except do_train will be True and output_dir will be the model path). (See: https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments)
    :param compute_metrics: (function = None), Function that calculates metrics. If None then accuracy, precision, recall and f1 scores are calculated
    :param callbacks: (Optional[List[TrainerCallback]] = None) A list of callbacks to customize the training loop. (see https://huggingface.co/transformers/main_classes/callback.html)
    :param optimizers: (Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None))) Will default to an instance of AdamW on your model and a scheduler given by get_linear_schedule_with_warmup() controlled by args.
    :return: model.
    """

    # dataset creation
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)

    # preparing the dataset for sequence classification
    ds, cls_to_index, index_to_cls = encode_dataset(tokenizer, data_loader,
                                                    label_all_subword_units=label_all_subword_units,
                                                    tokenization_args=tokenization_args,
                                                    map_args=map_args)
    if ds is None:
        raise Exception("No dataset was provided")

    model = BertForTokenClassification.from_pretrained(model_path, num_labels=len(cls_to_index))

    model.resize_token_embeddings(len(tokenizer))
    try:
        model.bert.embeddings.word_embeddings.padding_idx = 1
    except:
        print("Failed to reset padding idx")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_deprecation_fix(training_args)),
        train_dataset=ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,
    )

    trainer.train()

    if save_model:
        trainer.model.config.id2label = index_to_cls
        trainer.model.config.label2id = cls_to_index
        trainer.save_model()
        tokenizer.save_pretrained(training_args['output_dir'])

    return trainer.model


def evaluate(model_path: str,
             data_loader: Union[Tokens.Tsv, Tokens.EstNLTKCol],
             label_all_subword_units: bool = True,
             map_args: Optional[dict] = None,
             tokenizer_args: Optional[dict] = None,
             tokenization_args: Optional[dict] = None,
             training_args: dict = None,
             compute_metrics=None,
             callbacks: Optional[List[TrainerCallback]] = None,
             optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)):
    """
    Evaluates a bert model on a Token classification task
    :param model_path: (str) Path to the pretrained BERT model
    :param data_loader: (Union[Sequences.Tsv, Sequences.EstNLTKCol]) a class that is used to read the input data
    :param label_all_subword_units: (bool = True) to label all subword units or only the first
    :param map_args: (Optional[dict] = None) See: https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=map#datasets.Dataset.map
    :param tokenizer_args: (Optional[dict] = None) See: https://huggingface.co/transformers/model_doc/bert.html#berttokenizerfast
    :param tokenization_args: (Optional[dict] = None) See: https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
    :param training_args: (Optional[dict] = None), Arguments used in training. If None, the default training
         args will be used (except do_train will be True and output_dir will be the model path). (See: https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments)
    :param compute_metrics: (function = None), Function that calculates metrics. If None then accuracy, precision, recall and f1 scores are calculated
    :param callbacks: (Optional[List[TrainerCallback]] = None) A list of callbacks to customize the training loop. (see https://huggingface.co/transformers/main_classes/callback.html)
    :param optimizers: (Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None))) Will default to an instance of AdamW on your model and a scheduler given by get_linear_schedule_with_warmup() controlled by args.
    :return: The results of the compute metrics function + runtime stats.
    """

    # dataset creation
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)

    # preparing the dataset for sequence classification
    ds, cls_to_index, index_to_cls = encode_dataset(tokenizer, data_loader,
                                                    label_all_subword_units=label_all_subword_units,
                                                    tokenization_args=tokenization_args,
                                                    map_args=map_args)
    if ds is None:
        raise Exception("No dataset was provided")

    # a metric for sequence classification
    # can be replaced by providing a new function using the compute_metrics argument
    def tc_compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = []
        true_labels = []
        for prediction, label in zip(predictions, labels):
            for (p, l) in zip(prediction, label):
                if l != -100:
                    true_predictions.append(index_to_cls[p])
                    true_labels.append(index_to_cls[l])

        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average='weighted',
                                                                   zero_division=0)
        acc = accuracy_score(true_labels, true_predictions)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": acc,
        }

    model = BertForTokenClassification.from_pretrained(model_path, num_labels=len(cls_to_index))
    model.resize_token_embeddings(len(tokenizer))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_deprecation_fix(training_args)),
        eval_dataset=ds,
        compute_metrics=tc_compute_metrics if compute_metrics is None else compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,
    )

    return trainer.evaluate()


def encode_dataset(tokenizer, data_loader=None, label_all_subword_units=False, class_to_index=None,
                   tokenization_args=None, map_args=None):
    """
    Tokenizes the input datasets and makes sure that labels are turned into indices
    :param tokenizer: (transformers.PreTrainedTokenize) See: https://huggingface.co/transformers/main_classes/tokenizer.html#pretrainedtokenizer
    :param data_loader: (Union[Sequences.Tsv, Sequences.EstNLTKCol]) a class that is used to read the input data
    :param label_all_subword_units: (bool, default=False) if set to True, the function labels all subword units, if False, then only the first
    :param class_to_index: (dict, default=None) a dictionary that maps classes/labels to indices
    :param tokenization_args: (dict, default=None) note: is_split_into_words is set to True, original default = False". See: https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
    :param map_args: (dict, default=None) See: https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=map#datasets.Dataset.map
    :return:
    """

    # token rows into sentences

    dataset = data_loader.read()

    if class_to_index is None:
        # finding all class labels
        classes = set()
        for labels in dataset['y']:
            for lab in np.unique(labels):
                classes.add(lab)
        # converting classes set to a list and sorting it so it would be deterministic
        classes = list(classes)
        classes.sort()
        classes = [str(i) for i in classes]

        # assigning an index to a class and vice versa
        class_to_index = {v: k for k, v in enumerate(classes)}
        index_to_class = {k: v for k, v in enumerate(classes)}
    else:
        index_to_class = {v: k for k, v in class_to_index.items()}

    # 'is_split_into_words must be True
    if 'is_split_into_words' not in tokenization_args:
        tokenization_args['is_split_into_words'] = True

    # https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(text=examples["X"], **tokenization_args)

        labels = []
        for i, label in enumerate(examples["y"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(class_to_index[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(class_to_index[label[word_idx]] if label_all_subword_units else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return dataset.map(tokenize_and_align_labels, remove_columns=["X", "y"], **map_args), class_to_index, index_to_class
