import transformers
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import DataCollatorForLanguageModeling, Trainer, BertForNextSentencePrediction, BertForMaskedLM, \
    AutoTokenizer, TrainingArguments

from pipelines.step02_BERT_pre_training.pre_training.Helpers import training_args_deprecation_fix
from pipelines.step02_BERT_pre_training.tokenizing.text_dataset_for_NSP import create_dataset_for_NSP
import numpy as np


def eval_pretrained_BERT(model_path, input_files, nsp_probability=0.5, mlm_probability=0.15, tokenizer_args=None,
                         tokenization_args=None, training_args=None, callbacks=None, verbose=False):
    """
        :param model_path: (string) path to the model directory, if the directory does not exist, then it is created
        :param input_files: (string or [string]) path(s) to the input .tsv files
        :param mlm_probability: (float, default=0.15), the probability that a token is masked.
        :param nsp_probability: (float, default=0.5), the probability that the next sentence is consecutive.
        :param tokenizer_args:  (See: https://huggingface.co/transformers/model_doc/bert.html#berttokenizerfast)
        :param tokenization_args:  (See: https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__)
        :param training_args: (dict, default=None), Arguments used in training. If None, the default training
         args will be used (except do_train will be True and output_dir will be the model path). (See: https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments)
        :param callbacks: A list of callbacks to customize the training loop. (see https://huggingface.co/transformers/main_classes/callback.html)
        :param verbose: (boolean, default=False), displays metrics if True.
        :returns tuple(mlm_metrics, nsp_metric)
        """

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)
    dataset = load_dataset("csv", data_files={'train': input_files})['train']
    dataset_enc = create_dataset_for_NSP(dataset, tokenizer, nsp_probability=nsp_probability, **tokenization_args)

    def mlm_compute_metrics(pred):
        labels = pred.label_ids
        pred = np.argmax(pred.predictions, axis=-1)
        res = []
        lab = []
        for i, row in enumerate(labels):
            for j, el in enumerate(row):
                if el != -100:
                    res.append(pred[i][j])
                    lab.append(el)

        precision, recall, f1, _ = precision_recall_fscore_support(lab, res, average='weighted', zero_division=0)
        acc = accuracy_score(lab, res)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    # Evaluating on MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )

    model = BertForMaskedLM.from_pretrained(model_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_deprecation_fix(training_args)),
        eval_dataset=dataset_enc,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
        compute_metrics=mlm_compute_metrics
    )

    mlm_res = trainer.evaluate()
    if verbose:
        print(f"MLM metrics:{mlm_res}")

    # Evaluating on NSP
    def nsp_compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    model = BertForNextSentencePrediction.from_pretrained(model_path)
    dataset_enc = dataset_enc.rename_column("next_sentence_label", "labels")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_deprecation_fix(training_args)),
        eval_dataset=dataset_enc,
        tokenizer=tokenizer,
        callbacks=callbacks,
        compute_metrics=nsp_compute_metrics
    )

    nsp_res = trainer.evaluate()
    if verbose:
        print(f"NSP metrics:{nsp_res}")

    return mlm_res, nsp_res
