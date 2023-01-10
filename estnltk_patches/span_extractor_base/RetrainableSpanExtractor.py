from label_studio_ml.model import LabelStudioMLBase
from estnltk_patches.span_extractor_base.SpanExtractorBase import SpanExtractorBase
from typing import List
from estnltk import Tagger, Text
from estnltk_patches.taggers.ner_tagger.ner_tagger import BertNerTagger
from estnltk_patches.taggers.ner_tagger.DynamicTagger import DynamicTagger
from estnltk_patches.span_extractor_base.labelstudio_json_parser import PredictionResults, ResultsEntry
import requests
import random
import string
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RetrainableSpanExtractor(LabelStudioMLBase):

    def __init__(self, **kwargs):
        print("================================================")
        print(kwargs)
        print("================================================")
        super(RetrainableSpanExtractor, self).__init__(**kwargs)

        self.markup_taggers: List[Tagger] = self._initialize_markup_taggers()

        if self.train_output and 'finetuned_model_path' in self.train_output:
            self.retrainable_tagger: DynamicTagger = self._initialize_finetuned_tagger(
                pretrained_model_path=self.pretrained_model_path,
                finetuned_model_path=self.train_output['finetuned_model_path'])

        self.access_token = "d5d2155c8389ce27da7e75d5bfde36b297530b94"

        try:
            self.from_name, self.info = list(self.parsed_label_config.items())[0]
            self.to_name = self.info['to_name'][0]
            self.value = self.info['inputs'][0]['value']
        except IndexError:
            pass

        self.tagger_list = self.markup_taggers
        self.conf_html = self.conf_gen(self.tagger_list)

        print("Copy the generated configuration into labelstudio interface code part: \n")
        print(self.conf_html)

    def _initalize_retrainable_tagger(self, pretrained_model_path: str) -> DynamicTagger:
        raise NotImplementedError('Define a retrainable BERT tagger')

    def _initialize_finetuned_tagger(self, pretrained_model_path: str, finetuned_model_path: str) -> DynamicTagger:
        raise NotImplementedError('Define a fine-tuned tagger used for predicting annotations')

    def _initialize_markup_taggers(self) -> List[Tagger]:
        raise NotImplementedError('Define a list of taggers used for additional text markup')

    @property
    def pretrained_model_path(self):
        raise NotImplementedError("Define pretrained model path")

    # eraldi eksplisiitne funktsioon, mitte seotud selfi jms
    def predict(self, tasks, **kwargs):
        input_texts = []
        for task in tasks:
            input_texts.append(task['data'][self.value])

        predictions = []

        for sentence in input_texts:
            text = Text(sentence)

            for tagger in self.markup_taggers:
                tagger(text)

            if hasattr(self, "retrainable_tagger"):
                self.retrainable_tagger.tag(text)

            result = []

            for layers in text.list_layers():
                for span in layers.spans:
                    # print("layers:", layers)
                    for annotation in span.annotations:
                        # print("Annotation:", annotation.__dict__['grammar_symbol'])
                        # print("span:", annotation.span)
                        # print("start:", annotation.start)
                        # print("end:", annotation.end)
                        # print("text:", annotation.span.text)
                        result.append({
                            # id generaator eraldi funktsioonina
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
                                    layers.name if layers.name != 'ner' else annotation.__dict__['grammar_symbol']
                                ]
                            }
                        })
                predictions.append({'result': result, 'score': 0})
        # print("Predictions incoming ----------- \n \n")
        # print(predictions)
        # print("Predictions ending ----------- \n \n")
        print("We got predictions")
        return predictions

    def _get_annotated_dataset(self, project_id):
        """Just for demo purposes: retrieve annotated data from Label Studio API"""
        HOSTNAME = 'http://localhost:8080'
        download_url = f'{HOSTNAME.rstrip("/")}/api/projects/{project_id}/export'
        response = requests.get(download_url, headers={'Authorization': f'Token {self.access_token}'})
        if response.status_code != 200:
            raise Exception(f"Can't load task data using {download_url}, "
                            f"response status_code = {response.status_code}")
        return json.loads(response.content)

    # Eraldi eksplisiitne funktsioon samuti
    def fit(self, completions, workdir=None, **kwargs):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Got called to fit function")
        print("kwargs:", kwargs)
        print("completions:", completions)
        print("Access token stuff: \n")

        if kwargs.get('data') and kwargs.get('event') == 'PROJECT_UPDATED':
            project_id = kwargs['data']['project']['id']
            try:
                completions = self._get_annotated_dataset(project_id)
                print(completions)
            except:
                print("Couldnt get the annotated dataset \n \n \n XXXX \n \n \n")
            print("--- TASKS ---")

            # ... do some heavy computations, get your model and store checkpoints and resources
            annotations = []
            retrainable_tagger: DynamicTagger = self._initalize_retrainable_tagger(pretrained_model_path=self.pretrained_model_path)
            annotations = retrainable_tagger.annotations
            print("completions incoming -----------------------------")
            results = PredictionResults(tasks=[], bert_annotations=annotations, markup_taggers=self.markup_taggers)
            for completion in completions:
                print(completion)
                results.entries.append(ResultsEntry(entry=completion, bert_annotations=annotations))

            # print("results incoming -----------")
            # for sd in results.entries:
            #     print(sd)

            print("workdir:", workdir)
            print(self.train_output)
            print(self.info)

            finetune_training_data = os.path.join(workdir, 'stuckinthemiddlewithyou.tsv')
            results.make_training_file(finetune_training_data)

            self.model_version = str(workdir)

            # time.sleep(10)
            # print("self.tagger_list:", self.tagger_list)
            # for tagger in self.tagger_list:
            #     if isinstance(tagger, BertNerTagger):
            #         tagger.train(training_data=finetune_training_data,
            #                      finetuned_model_path=os.path.join(workdir, 'token_classifier_model'))
            retrainable_tagger.train(training_data=finetune_training_data,
                                     finetuned_model_path=os.path.join(workdir, 'token_classifier_model'))

            return {'finetune_training_data_path': finetune_training_data,
                    'finetuned_model_path': os.path.join(workdir, 'token_classifier_model'),
                    'version': workdir}
        return {}

    def conf_gen(self, tagger_list: List[Tagger]):
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
