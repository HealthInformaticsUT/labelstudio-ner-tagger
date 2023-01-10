from estnltk import Layer, Text, Span, BaseSpan, ElementaryBaseSpan, Tagger
from estnltk.taggers.text_segmentation.tokens_tagger import TokensTagger
import csv
from typing import Dict
from typing import AnyStr
from typing import List

# from estnltk_patches.taggers.robust_date_number_tagger.robust_date_number_tagger import RobustDateNumberTagger
# from estnltk_patches.taggers.cancer_stage_tagger.cancer_stage_tagger import CancerStageTagger
#
#
# tagger1 = RobustDateNumberTagger()
# tagger2 = CancerStageTagger()
#
# taggers = [tagger1, tagger2]


class PredictionResults:
    def __init__(self, tasks, bert_annotations: List[AnyStr], markup_taggers: List[Tagger]):
        self.tasks = tasks
        self.entries = []
        self.bert_annotations = bert_annotations
        self.markup_tagger_mapping = {}

        for enum, markup_tagger in enumerate(markup_taggers, start=1):
            self.markup_tagger_mapping[markup_tagger.output_layer] = '[UNK{number}]'.format(number=str(enum))

        print(self.markup_tagger_mapping)

        for task in self.tasks:
            results_entry = ResultsEntry(entry=task, bert_annotations=self.bert_annotations)
            self.entries.append(results_entry)

    def make_training_file(self, file_path: str):
        f_finetuning_training_data = open(file_path, 'w')
        writer = csv.writer(f_finetuning_training_data, delimiter='\t')
        writer.writerow(['text', 'y'])

        for entry in self.entries:
            for k, v in entry.tokenized().items():
                word = v[0]
                annotations = v[1]
                current_bert_annot = None
                # print(word, annotations)
                for annotation in annotations:
                    if annotation in self.bert_annotations:
                        current_bert_annot = annotation
                    elif annotation in self.markup_tagger_mapping:
                        word = self.markup_tagger_mapping[annotation]
                if current_bert_annot is not None:
                    writer.writerow([word, current_bert_annot])
                else:
                    writer.writerow([word, "0"])
                # writer.writerow([word, "0"])
            writer.writerow([""])

        f_finetuning_training_data.close()


class ResultsEntry:
    def __init__(self, entry: Dict[AnyStr, List], bert_annotations: List[AnyStr]):
        self.raw_text = entry['data']['text']
        self.text = Text(entry['data']['text'])
        self.id = entry['id']
        self.choice = None
        self.bert_annotations = bert_annotations

        for span in entry['annotations'][0]['result']:
            if 'choices' in span['value']:
                if span['value']['choices'][0] == 'good':
                    self.choice = True
                elif span['value']['choices'][0] == 'bad':
                    self.choice = False

            elif 'labels' in span['value']:
                start = span['value']['start']
                end = span['value']['end']
                span_text = span['value']['text']
                labels = span['value']['labels']

                for label in labels:
                    if label in self.bert_annotations:
                        # print("checkpoint 1", label, span_text)
                        new_layer = Layer(name='ner', attributes=['grammar_symbol', 'value'], text_object=None, ambiguous=True)
                    else:
                        # print("checkpoint 2", label, span_text)
                        new_layer = Layer(name=label, attributes=['grammar_symbol', 'value'], text_object=None, ambiguous=True)
                    if new_layer.name in self.text.layers:
                        # print("checkpoint 3", label, span_text)
                        layer_name = 'ner' if label in self.bert_annotations else label
                        self.text[layer_name].add_annotation(
                            ElementaryBaseSpan(
                                start=int(start),
                                end=int(end)),
                            **{'grammar_symbol': label, 'value': span_text})
                    elif new_layer.name not in self.text.layers:
                        # print("checkpoint 4", label, span_text)
                        new_layer.add_annotation(
                            ElementaryBaseSpan(
                                start=int(start),
                                end=int(end)),
                            **{'grammar_symbol': label, 'value': span_text}
                        )

                        self.text.add_layer(new_layer)

    def tokenized(self):
        return tokenize_for_training(self.text)


def tokenize_for_training(text: Text):
    estnltk_tokenizer = TokensTagger()
    estnltk_tokenizer.tag(text)

    tokens_and_spans = {(span.start, span.end): [span.text, set()] for span in text.tokens}

    for layer in text.list_layers():
        if layer.name != 'tokens':
            for span in layer:
                grammar_symbol = span.annotations[0]['grammar_symbol']
                start = span.start
                end = span.end

                for k, v in tokens_and_spans.items():
                    start_ = k[0]
                    end_ = k[1]

                    if len(range(max(start, start_), min(end, end_))) > 0 and len(grammar_symbol) > 0:
                        tokens_and_spans[k][1] = v[1].union([grammar_symbol])

    return tokens_and_spans


# data = [{'id': 822, 'annotations': [{'id': 93, 'completed_by': 1, 'result': [
#     {"value": {"choices": ["good"]}, "id": "5T-w7gAwjL", "from_name": "review", "to_name": "text", "type": "choices",
#      "origin": "manual"},
#     {"value": {"start": 39, "end": 51, "text": "depressiooni", "labels": ["B-SYMP"]}, "id": "MyYyHuboiz",
#      "from_name": "label", "to_name": "text", "type": "labels", "origin": "prediction"},
#     {'value': {'start': 27, 'end': 32, 'text': '2008a', 'labels': ['dates_numbers']}, 'id': 'gvDXKolt6s',
#      'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'},
#     {'value': {'start': 62, 'end': 63, 'text': '4', 'labels': ['dates_numbers']}, 'id': 'HUKZPFELOv',
#      'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'},
#     {'value': {'start': 72, 'end': 74, 'text': '10', 'labels': ['dates_numbers']}, 'id': '7B8c5CwReY',
#      'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'},
#     {'value': {'start': 69, 'end': 73, 'text': 'st 1', 'labels': ['stages']}, 'id': 'UPknXMsTWD', 'from_name': 'label',
#      'to_name': 'text', 'type': 'labels', 'origin': 'prediction'},
#     {'value': {'start': 73, 'end': 76, 'text': '0st', 'labels': ['stages']}, 'id': 'z9vm9HP8Dy', 'from_name': 'label',
#      'to_name': 'text', 'type': 'labels', 'origin': 'prediction'},
#     {'value': {'start': 83, 'end': 87, 'text': 'IIA ', 'labels': ['stages']}, 'id': 'NHSHNBzZ25', 'from_name': 'label',
#      'to_name': 'text', 'type': 'labels', 'origin': 'prediction'}], 'was_cancelled': False, 'ground_truth': False,
#                                      'created_at': '2022-02-07T07:45:09.857634Z',
#                                      'updated_at': '2022-02-07T07:45:09.857671Z', 'lead_time': 1.208, 'prediction': {},
#                                      'result_count': 0, 'task': 822, 'parent_prediction': 580,
#                                      'parent_annotation': None}], 'file_upload': '520c36a2-testmultipletaggers.tsv',
#          'drafts': [], 'predictions': [580], 'data': {
#         'text': 'Minu vanemad kasvasid üles 2008a aasta depressiooni ajal, kus 4 inimest 10st surid IIA kasvajasse'},
#          'meta': {}, 'created_at': '2022-01-26T09:09:51.583992Z', 'updated_at': '2022-01-26T09:09:51.584031Z',
#          'project': 7}, {'id': 823, 'annotations': [{'id': 94, 'completed_by': 1, 'result': [
#     {'value': {'start': 0, 'end': 4, 'text': 'IIA ', 'labels': ['stages']}, 'id': 'qAqpTYKB6g', 'from_name': 'label',
#      'to_name': 'text', 'type': 'labels', 'origin': 'prediction'}], 'was_cancelled': False, 'ground_truth': False,
#                                                      'created_at': '2022-02-07T07:45:11.817270Z',
#                                                      'updated_at': '2022-02-07T07:45:11.817317Z', 'lead_time': 0.786,
#                                                      'prediction': {}, 'result_count': 0, 'task': 823,
#                                                      'parent_prediction': 581, 'parent_annotation': None}],
#                          'file_upload': '4ab7e429-testmultipletaggers.tsv', 'drafts': [], 'predictions': [581],
#                          'data': {'text': 'IIA vähk ei ole mingi naljaasi'}, 'meta': {},
#                          'created_at': '2022-02-01T13:01:06.718459Z', 'updated_at': '2022-02-01T13:01:06.718489Z',
#                          'project': 7}, {'id': 826, 'annotations': [{'id': 92, 'completed_by': 1, 'result': [
#     {'value': {'start': 55, 'end': 63, 'text': '29.04.10', 'labels': ['dates_numbers']}, 'id': 'c5qnRJzcwS',
#      'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'},
#     {'value': {'start': 91, 'end': 92, 'text': '2', 'labels': ['dates_numbers']}, 'id': 'KLpoosE4HN',
#      'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'},
#     {'value': {'start': 93, 'end': 94, 'text': '3', 'labels': ['dates_numbers']}, 'id': 'GzYa7ZlbJW',
#      'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'},
#     {'value': {'start': 95, 'end': 96, 'text': '0', 'labels': ['dates_numbers']}, 'id': 'kh6gNfgrq0',
#      'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'},
#     {'value': {'start': 114, 'end': 124, 'text': '26.03.2011', 'labels': ['dates_numbers']}, 'id': 'vLlSWXyFMb',
#      'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'},
#     {'value': {'start': 244, 'end': 246, 'text': '11', 'labels': ['dates_numbers']}, 'id': '7kfjyeHqbg',
#      'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'},
#     {'value': {'start': 265, 'end': 270, 'text': '2012a', 'labels': ['dates_numbers']}, 'id': 'HSTLoGG8Ee',
#      'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'},
#     {'value': {'start': 89, 'end': 96, 'text': ' T2N3M0', 'labels': ['B-SYMP']}, 'id': 'KNTXl3m4ih',
#      'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'},
#     {'value': {'start': 172, 'end': 176, 'text': 'IIA ', 'labels': ['stages']}, 'id': 'Fm2FjNxK2D',
#      'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'}], 'was_cancelled': False,
#                                                                      'ground_truth': False,
#                                                                      'created_at': '2022-02-07T07:45:04.301801Z',
#                                                                      'updated_at': '2022-02-07T07:45:04.301835Z',
#                                                                      'lead_time': 9.777, 'prediction': {},
#                                                                      'result_count': 0, 'task': 826,
#                                                                      'parent_prediction': 584,
#                                                                      'parent_annotation': None}],
#                                          'file_upload': '4ab7e429-testmultipletaggers.tsv', 'drafts': [],
#                                          'predictions': [584], 'data': {
#         'text': 'Patsiendil esines eelmisel külastusel, mis leidis aset 29.04.10 tõsiseid tüsistusi seoses T2N3M0 kasvajaga kõhus. 26.03.2011 andis ta teada, et enam ei taha elada. Eks see IIA vähk ei ole tore asi. Patsiendist jääb järele koer Isabella, kes on 11 aastane kutsikas. 2012a tahaks talle kodu leida.'},
#                                          'meta': {}, 'created_at': '2022-02-01T13:01:06.718639Z',
#                                          'updated_at': '2022-02-07T07:03:02.722761Z', 'project': 7}, {'id': 827,
#                                                                                                       'annotations': [
#                                                                                                           {'id': 95,
#                                                                                                            'completed_by': 1,
#                                                                                                            'result': [{
#                                                                                                                'value': {
#                                                                                                                    'start': 27,
#                                                                                                                    'end': 32,
#                                                                                                                    'text': '2008a',
#                                                                                                                    'labels': [
#                                                                                                                        'dates_numbers']},
#                                                                                                                'id': 'C3ZSrJ3Nuw',
#                                                                                                                'from_name': 'label',
#                                                                                                                'to_name': 'text',
#                                                                                                                'type': 'labels',
#                                                                                                                'origin': 'prediction'},
#                                                                                                                {
#                                                                                                                    'value': {
#                                                                                                                        'start': 62,
#                                                                                                                        'end': 63,
#                                                                                                                        'text': '4',
#                                                                                                                        'labels': [
#                                                                                                                            'dates_numbers']},
#                                                                                                                    'id': 'yqLtSkOTJh',
#                                                                                                                    'from_name': 'label',
#                                                                                                                    'to_name': 'text',
#                                                                                                                    'type': 'labels',
#                                                                                                                    'origin': 'prediction'},
#                                                                                                                {
#                                                                                                                    'value': {
#                                                                                                                        'start': 72,
#                                                                                                                        'end': 74,
#                                                                                                                        'text': '10',
#                                                                                                                        'labels': [
#                                                                                                                            'dates_numbers']},
#                                                                                                                    'id': 'gieLXbyTUI',
#                                                                                                                    'from_name': 'label',
#                                                                                                                    'to_name': 'text',
#                                                                                                                    'type': 'labels',
#                                                                                                                    'origin': 'prediction'},
#                                                                                                                {
#                                                                                                                    'value': {
#                                                                                                                        'start': 69,
#                                                                                                                        'end': 73,
#                                                                                                                        'text': 'st 1',
#                                                                                                                        'labels': [
#                                                                                                                            'stages']},
#                                                                                                                    'id': 'tbQABAgngr',
#                                                                                                                    'from_name': 'label',
#                                                                                                                    'to_name': 'text',
#                                                                                                                    'type': 'labels',
#                                                                                                                    'origin': 'prediction'},
#                                                                                                                {
#                                                                                                                    'value': {
#                                                                                                                        'start': 73,
#                                                                                                                        'end': 76,
#                                                                                                                        'text': '0st',
#                                                                                                                        'labels': [
#                                                                                                                            'stages']},
#                                                                                                                    'id': 'f8ZSjdJKMS',
#                                                                                                                    'from_name': 'label',
#                                                                                                                    'to_name': 'text',
#                                                                                                                    'type': 'labels',
#                                                                                                                    'origin': 'prediction'},
#                                                                                                                {
#                                                                                                                    'value': {
#                                                                                                                        'start': 83,
#                                                                                                                        'end': 87,
#                                                                                                                        'text': 'IIA ',
#                                                                                                                        'labels': [
#                                                                                                                            'stages']},
#                                                                                                                    'id': 'mF5qa7lGqL',
#                                                                                                                    'from_name': 'label',
#                                                                                                                    'to_name': 'text',
#                                                                                                                    'type': 'labels',
#                                                                                                                    'origin': 'prediction'}],
#                                                                                                            'was_cancelled': False,
#                                                                                                            'ground_truth': False,
#                                                                                                            'created_at': '2022-02-07T07:45:17.656135Z',
#                                                                                                            'updated_at': '2022-02-07T07:45:17.656172Z',
#                                                                                                            'lead_time': 1.369,
#                                                                                                            'prediction': {},
#                                                                                                            'result_count': 0,
#                                                                                                            'task': 827,
#                                                                                                            'parent_prediction': 586,
#                                                                                                            'parent_annotation': None}],
#                                                                                                       'file_upload': '4ab7e429-testmultipletaggers.tsv',
#                                                                                                       'drafts': [],
#                                                                                                       'predictions': [
#                                                                                                           586],
#                                                                                                       'data': {
#                                                                                                           'text': 'Minu vanemad kasvasid üles 2008a aasta depressiooni ajal, kus 4 inimest 10st surid IIA kasvajasse'},
#                                                                                                       'meta': {},
#                                                                                                       'created_at': '2022-02-01T13:01:06.718686Z',
#                                                                                                       'updated_at': '2022-02-01T13:01:06.718698Z',
#                                                                                                       'project': 7}]

# results = PredictionResults(tasks=data, bert_annotations=['B-SYMP', 'B-DRUG'], markup_taggers=taggers)
# results.make_training_file(file_path='fine_tuning_training2.tsv')


