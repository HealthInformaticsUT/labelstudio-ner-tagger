from typing import MutableMapping
from estnltk.taggers import Tagger
from estnltk import Layer, Text, Span, BaseSpan, ElementaryBaseSpan

from pipelines.step03_BERT_fine_tuning import TokenClassification
from pipelines.step03_BERT_fine_tuning.dataloaders import Tokens

from transformers import AutoTokenizer, BertForTokenClassification, logging
from transformers import pipeline

from estnltk_patches.taggers.ner_tagger.DynamicTagger import DynamicTagger

import os


class BertNerTagger(DynamicTagger):
    """
    """

    conf_param = ['pretrained_model_path', 'tokenizer_args', 'tokenization_args', 'tokenizer', 'model', 'pipeline', 'annotations', 'finetuned_model_path']

    # @TODO - Correct relative pathing to model

    def __init__(
            self,
            pretrained_model_path=None,
            output_layer='ner',
            annotations=None,
            finetuned_model_path=None
    ):
        print("\n ---------- WE ARE INITING A NEW NER TAGGER ---------- \n")

        if pretrained_model_path is None:
            self.pretrained_model_path = '../../../medbert_models/pretr_copy'
        else:
            self.pretrained_model_path = pretrained_model_path

        self.tokenizer_args = {
            "lowercase": False
        }
        self.tokenization_args = {
            "max_length": 128,
            "padding": "max_length",
            "truncation": True,
            'is_split_into_words': True
        }

        self.finetuned_model_path = finetuned_model_path
        self.pipeline = None
        if self.finetuned_model_path is not None:
            if os.path.isdir(self.finetuned_model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(self.finetuned_model_path, **self.tokenizer_args)
                self.model = BertForTokenClassification.from_pretrained(self.finetuned_model_path)
                self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

        self.output_layer = output_layer
        self.output_attributes = ['grammar_symbol', 'value']
        self.annotations = annotations
        self.input_layers = []

        # self.tokenizer, self.model, self.pipeline = None, None, None

        # if hasattr(self, 'finetuned_model_path'):
        #     if os.path.isdir(self.finetuned_model_path):
        #         self.tokenizer = AutoTokenizer.from_pretrained(self.finetuned_model_path, **self.tokenizer_args)
        #         self.model = BertForTokenClassification.from_pretrained(self.finetuned_model_path)
        #         self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def _make_layer_template(self):
        """Creates and returns a template of the layer."""
        return Layer(name=self.output_layer, attributes=self.output_attributes, text_object=None)

    def _make_layer(self, text: Text, layers: MutableMapping[str, Layer], status: dict) -> Layer:

        nerlayer = self._make_layer_template()
        nerlayer.text_object = text

        print("\n ------------ NER TAGGER IS TRYING TO DO THINGS -------- \n")
        print("self.pipeline:", self.pipeline)

        if self.pipeline is not None:

            print("\n ------------ NER TAGGER IS DOING THINGS -------- \n")
            print("We are currently using", self.finetuned_model_path, "model")
            tag_results = self.pipeline(text.text)
            spanned_results = retrieve_spans_from_ner(tag_results)
            # print(spanned_results)

            # {'entity': '0', 'score': 0.9950411, 'index': 1, 'word': 'patsiendil', 'start': 0, 'end': 10}

            for result in spanned_results:
                # base_span = BaseSpan(raw=result['entity'], level=0, start=result['start'], end=result['end'])
                # print(base_span)
                # print("Ner tagger specifics incoming ---------- \n")
                # print("We are currently getting: ")
                # print("Span is ", result['start'], result['end'], result['word'])
                # print("But we could also get for word:")
                # print(text.text[result['start']:result['end']])

                # print("Result:", result)
                # print("Result attributes:", result['entity'], result['word'], result['start'], result['end']) # B-DRUG dur 63 66
                if str(result['entity']) != '0':
                    print("- got a ner tag -")
                    print(result['entity'], result['word'])
                    nerlayer.add_annotation(
                        ElementaryBaseSpan(start=int(result['start']), end=int(result['end'])),
                        **{self.output_attributes[0]: result['entity'], self.output_attributes[1]: result['word']}
                    )

        return nerlayer
        # return self.tagger.make_layer(text=text, layers=layers, status=status)

    def train(self, training_data: str, finetuned_model_path: str):

        if finetuned_model_path is not None:
            self.finetuned_model_path = finetuned_model_path
            # super(BertNerTagger, self).__setattr__("finetuned_model_path", finetuned_model_path)
            # self.finetuned_model_path = finetuned_model_path

        if not os.path.isdir(self.finetuned_model_path):
            dl = Tokens.Tsv([training_data])

            print("finetuned model path:", self.finetuned_model_path)
            print("training_data path:", training_data)
            print("dl incoming:")
            print(dl)
            print(dl.read())

            if dl.read().num_rows > 0:
                map_args = {
                    "batched": True
                }

                training_args = {
                    "output_dir": self.finetuned_model_path,
                    "overwrite_output_dir": True,
                    "num_train_epochs": 1,
                    "per_device_train_batch_size": 8,
                    "per_device_eval_batch_size": 8
                }

                TokenClassification.finetune_BERT(self.pretrained_model_path, dl, False, True, map_args, self.tokenizer_args,
                                                  self.tokenization_args, training_args)

                res = TokenClassification.evaluate(self.pretrained_model_path, dl, False, map_args, self.tokenizer_args,
                                                   self.tokenization_args, training_args)

                self.tokenizer = AutoTokenizer.from_pretrained(self.finetuned_model_path, **self.tokenizer_args)
                self.model = BertForTokenClassification.from_pretrained(self.finetuned_model_path)
                self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
                print("Checking for sanity...")
                print("Sanity found" if os.path.isdir(self.finetuned_model_path) else "Sanity not found")
                print("Saved fine-tuned model to path:", self.finetuned_model_path)
            else:
                print("--- 0 data given, no model trained ---")


def retrieve_spans_from_ner(ner_results: str):
    total = len(ner_results)
    counter = 0
    results = []

    while counter < total:
        spanstart = ner_results[counter]['start']
        spanend = ner_results[counter]['end']
        word = ner_results[counter]['word']
        entity = ner_results[counter]['entity']

        nextword = word

        for i in range(counter+1, total):
            if ner_results[i]['word'].startswith('##'):
                nextword += ner_results[i]['word']
                spanend = ner_results[i]['end']
                if ner_results[i]['entity'] != '0' and entity == '0':
                    entity = ner_results[i]['entity']
                counter += 1
            else:
                break
        results.append({'word': nextword.replace("##", ""), 'start': spanstart, 'end': spanend, 'entity': entity})
        counter += 1

    return results
