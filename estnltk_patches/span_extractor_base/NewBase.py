from typing import List

from label_studio_ml.model import LabelStudioMLBase
from estnltk import Text, Layer, Span, Tagger

import random
import string


class BaseWithTraining(LabelStudioMLBase):
    def __init__(self, tagger_list: List[Tagger], medbert_output_layers: List[str] = None, **kwargs):
        # don't forget to initialize base class...
        super(BaseWithTraining, self).__init__(**kwargs)

        print("config_items:", self.parsed_label_config.items())

        try:
            self.from_name, self.info = list(self.parsed_label_config.items())[0]
            self.to_name = self.info['to_name'][0]
            self.value = self.info['inputs'][0]['value']
        except IndexError:
            pass

        self.tagger_list = tagger_list
        self.medbert_output_layers = medbert_output_layers or []

        self.conf_html = conf_gen(self.tagger_list, self.medbert_output_layers)

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
                                    annotation.__dict__['grammar_symbol'] if layers.name=='ner' else layers.name
                                ]
                            }
                        })
                predictions.append({'result': result, 'score': 0})
        return predictions

    def fit(self, tasks, workdir=None, **kwargs):
        print("Fit function: ------------------ \n")
        print([task for task in tasks])
        print("Fit function end: ---------------- \n")
        return None


def conf_gen(tagger_list: List[Tagger], medbert_output_layers: List[str]):
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
        if tagger.output_layer == 'ner':
            continue
        conf_string += single_label.format(
            label_value=tagger.output_layer,
            background_value=("#" + "%06x" % random.randint(0, 0xFFFFFF)).upper()
        )

    for output_layer in medbert_output_layers:
        conf_string += single_label.format(
            label_value=output_layer,
            background_value=("#" + "%06x" % random.randint(0, 0xFFFFFF)).upper()
        )
    conf_string += end_block

    return conf_string
