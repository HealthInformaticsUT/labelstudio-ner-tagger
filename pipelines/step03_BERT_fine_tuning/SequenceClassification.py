import torch
import numpy as np

from typing import Optional, Tuple, List, Union

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainerCallback, TrainingArguments, \
    BertConfig

from pipelines.step02_BERT_pre_training.pre_training.Helpers import training_args_deprecation_fix
from pipelines.step03_BERT_fine_tuning.dataloaders import Sequences


def finetune_BERT(model_path: str,
                  data_loader: Union[Sequences.Tsv, Sequences.EstNLTKCol],
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
    ds, cls_to_index, index_to_cls = encode_dataset(tokenizer,
                                                    data_loader=data_loader,
                                                    tokenization_args=tokenization_args,
                                                    map_args=map_args)
    if ds is None:
        raise Exception("No dataset was provided")

    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(cls_to_index))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_deprecation_fix(training_args)),
        train_dataset=ds,
        compute_metrics=_sc_compute_metrics if compute_metrics is None else compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,
    )
    trainer.train()

    if save_model:
        trainer.model.config.id2label = index_to_cls
        trainer.model.config.label2id = cls_to_index
        trainer.save_model()
        tokenizer.save_pretrained(training_args['output_dir'])

    return model


def evaluate(model_path: str,
             data_loader: Union[Sequences.Tsv, Sequences.EstNLTKCol],
             map_args: Optional[dict] = None,
             tokenizer_args: Optional[dict] = None,
             tokenization_args: Optional[dict] = None,
             training_args: dict = None,
             compute_metrics=None,
             callbacks: Optional[List[TrainerCallback]] = None,
             optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)):
    """
    Evaluates a bert model on a sequence classification task
    :param model_path: (str) Path to the pretrained BERT model
    :param data_loader: (Union[Sequences.Tsv, Sequences.EstNLTKCol]) a class that is used to read the input data
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

    # try to load class_to_index from the config
    class_to_index = BertConfig.from_pretrained(model_path).label2id

    # preparing the dataset for sequence classification
    ds, cls_to_index, index_to_cls = encode_dataset(tokenizer,
                                                    data_loader=data_loader,
                                                    class_to_index=class_to_index,
                                                    tokenization_args=tokenization_args,
                                                    map_args=map_args)
    if ds is None:
        raise Exception("No dataset was provided")

    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(cls_to_index))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_deprecation_fix(training_args)),
        eval_dataset=ds,
        compute_metrics=_sc_compute_metrics if compute_metrics is None else compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,
    )
    return trainer.evaluate()


def encode_dataset(tokenizer,
                   data_loader: Union[Sequences.Tsv, Sequences.EstNLTKCol],
                   class_to_index=None,
                   tokenization_args=None,
                   map_args=None):
    """
    Tokenizes the input datasets and makes sure that labels are turned into indices
    :param tokenizer: (transformers.PreTrainedTokenize) See: https://huggingface.co/transformers/main_classes/tokenizer.html#pretrainedtokenizer
    :param data_loader: (Union[Sequences.Tsv, Sequences.EstNLTKCol]) a class that is used to read the input data
    :param class_to_index: (dict, default=None) a dictionary that maps classes/labels to indices
    :param tokenization_args: (dict, default=None) See: https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
    :param map_args: (dict, default=None) See: https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=map#datasets.Dataset.map
    :return:
    """

    dataset = data_loader.read()

    if class_to_index is None:
        # finding all class labels
        classes = set()
        for i in dataset["y"]:
            classes.add(i)
        # converting classes set to a list and sorting it so it would be deterministic
        classes = list(classes)
        classes.sort()
        classes = [str(i) for i in classes]

        # assigning an index to a class and vice versa
        class_to_index = {v: k for k, v in enumerate(classes)}
        index_to_class = {k: v for k, v in enumerate(classes)}
    else:
        index_to_class = {v: k for k, v in class_to_index.items()}

    # tokenization function
    def tokenize(batch):
        tokenized_sentences = tokenizer(text=batch["X"], **tokenization_args)
        tokenized_sentences['label'] = [class_to_index[str(i)] for i in batch["y"]]
        return tokenized_sentences

    return dataset.map(tokenize, remove_columns=["X", "y"], **map_args), class_to_index, index_to_class


# a metric for sequence classification
# can be replaced by providing a new function using the compute_metrics argument
def _sc_compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': float(acc), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}
