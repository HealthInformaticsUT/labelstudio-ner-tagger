from typing import List

from label_studio_ml.model import LabelStudioMLBase
from estnltk import Text, Layer, Span, Tagger

from estnltk_patches.taggers.ner_tagger.ner_tagger import BertNerTagger

from estnltk_patches.span_extractor_base.labelstudio_json_parser import PredictionResults, ResultsEntry

import random
import string
import os

# import time


class SpanExtractorBase(LabelStudioMLBase):
    def __init__(self, tagger_list: List[Tagger], **kwargs):
        # don't forget to initialize base class...
        super(SpanExtractorBase, self).__init__(**kwargs)

        print("config_items:", self.parsed_label_config.items())
        self.access_token = "d5d2155c8389ce27da7e75d5bfde36b297530b94"

        try:
            self.from_name, self.info = list(self.parsed_label_config.items())[0]
            self.to_name = self.info['to_name'][0]
            self.value = self.info['inputs'][0]['value']
        except IndexError:
            pass

        self.tagger_list = tagger_list
        self.conf_html = conf_gen(tagger_list)

        print("Copy the generated configuration into labelstudio interface code part: \n")
        print(self.conf_html)

    def predict(self, tasks, **kwargs):
        input_texts = []
        for task in tasks:
            input_texts.append(task['data'][self.value])

        predictions = []

        for sentence in input_texts:
            text = Text(sentence)

            for tagger in self.tagger_list:
                tagger(text)

            result = []

            for layers in text.list_layers():
                for span in layers.spans:
                    print("layers:", layers)
                    for annotation in span.annotations:
                        # print("Annotation:", annotation.__dict__['grammar_symbol'])
                        # print("span:", annotation.span)
                        # print("start:", annotation.start)
                        # print("end:", annotation.end)
                        # print("text:", annotation.span.text)
                        result.append({
                            'id': ''.join(
                                random.SystemRandom().choice(
                                    string.ascii_uppercase +
                                    string.ascii_lowercase +
                                    string.digits
                                )
                                for _ in
                                range(10)),
                            'from_name': self.from_name,
                            'to_name': self.to_name,
                            'type': 'labels',
                            'value': {
                                "start": annotation.start,
                                "end": annotation.end,
                                "text": annotation.span.text,
                                "labels": [
                                    # annotation.__dict__['grammar_symbol'] # nt. "stages",
                                    layers.name
                                ]
                            }
                        })
                predictions.append({'result': result, 'score': 0})
        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        # ... do some heavy computations, get your model and store checkpoints and resources
        annotations = []
        for tagger in self.tagger_list:
            if isinstance(tagger, BertNerTagger):
                annotations = tagger.annotations
        print("completions incoming -----------------------------")
        results = PredictionResults(tasks=[], bert_annotations=annotations)
        for completion in completions:
            print(completion)
            results.entries.append(ResultsEntry(entry=completion, bert_annotations=annotations))

        print("results incoming -----------")
        for sd in results.entries:
            print(sd)

        print("workdir:", workdir)
        print(self.train_output)
        print(self.info)

        finetune_training_data = os.path.join(workdir, 'stuckinthemiddlewithyou.tsv')
        results.make_training_file(finetune_training_data)

        # time.sleep(10)
        print("self.tagger_list:", self.tagger_list)
        for tagger in self.tagger_list:
            if isinstance(tagger, BertNerTagger):
                tagger.train(training_data=finetune_training_data, finetuned_model_path=os.path.join(workdir, 'token_classifier_model'))

        return {'finetune_training_data_path': finetune_training_data,
                'finetuned_model_path': os.path.join(workdir, 'token_classifier_model')}  # <-- you can retrieve this dict as self.train_output in the subsequent calls


def conf_gen(tagger_list: List[Tagger]):
    single_label = '\t<Label value="{label_value}" background="{background_value}"/> \n'
    conf_string = """
<View>
    <Labels name="label" toName="text">\n"""
    end_block = """
    </Labels>
<Text name="text" value="$text"/>
<Header value="Are the annotations correct?"/>
<Choices name="review" toName="text">
    <Choice value="good"/>
    <Choice value="bad"/>
</Choices>
</View>"""

    for tagger in tagger_list:
        conf_string += single_label.format(
            label_value=tagger.output_layer,
            background_value=("#" + "%06x" % random.randint(0, 0xFFFFFF)).upper()
        )
    conf_string += end_block

    return conf_string
