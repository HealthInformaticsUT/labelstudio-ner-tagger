import unittest
import csv
import os

from ..span_extractor_base.labelstudio_json_parser import PredictionResults


class TestLabels(unittest.TestCase):

    def test_ner_annotations(self):

        test_file = 'estnltk_patches/tests/test_training_data1.tsv'

        data = [
            {'id': 1, 'annotations': [
                {'id': 93, 'completed_by': 1, 'result': [
                    {"value":
                         {"choices": ["good"]}, "id": "5T-w7gAwjL", "from_name": "review", "to_name": "text", "type": "choices",
                     "origin": "manual"},

                    {"value":
                         {"start": 5, "end": 10, "text": "xanax", "labels": ["B-DRUG"]}, "id": "test1",
                     "from_name": "label", "to_name": "text", "type": "labels", "origin": "prediction"},
                    {'value':

                         {'start': 16, 'end': 23, 'text': 'peavalu', 'labels': ['B-SYMP']}, 'id': 'test2',
                     'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'}],
                    }],

                'data': {
                    'text': 'test xanax test peavalu'},
                 }]

        results = PredictionResults(tasks=data, bert_annotations=['B-SYMP', 'B-DRUG'])
        results.make_training_file(file_path=test_file)

        control_dict = {}

        with open(test_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, line in enumerate(reader):
                control_dict[i] = line

        self.assertEqual(control_dict[2], ['xanax', 'B-DRUG'])
        self.assertEqual(control_dict[4], ['peavalu', 'B-SYMP'])

        os.remove(test_file)

    def test_classifications(self):
        data = [
            {'id': 1, 'annotations': [
                {'id': 93, 'completed_by': 1, 'result': [
                    {"value":
                         {"choices": ["good"]}, "id": "5T-w7gAwjL", "from_name": "review", "to_name": "text",
                     "type": "choices",
                     "origin": "manual"},

                    {"value":
                         {"start": 5, "end": 10, "text": "xanax", "labels": ["B-DRUG"]}, "id": "test1",
                     "from_name": "label", "to_name": "text", "type": "labels", "origin": "prediction"},
                    {'value':

                         {'start': 16, 'end': 23, 'text': 'peavalu', 'labels': ['B-SYMP']}, 'id': 'test2',
                     'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'}],
                 }],

             'data': {
                 'text': 'test xanax test peavalu'},
             },
            {'id': 2, 'annotations': [
                {'id': 94, 'completed_by': 1, 'result': [
                    {"value":
                         {"choices": ["bad"]}, "id": "5T-w7gAwjL", "from_name": "review", "to_name": "text",
                     "type": "choices",
                     "origin": "manual"},

                    {"value":
                         {"start": 5, "end": 10, "text": "xanax", "labels": ["B-DRUG"]}, "id": "test1",
                     "from_name": "label", "to_name": "text", "type": "labels", "origin": "prediction"},
                    {'value':

                         {'start': 16, 'end': 23, 'text': 'peavalu', 'labels': ['B-SYMP']}, 'id': 'test2',
                     'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'}],
                 }],

             'data': {
                 'text': 'test xanax test peavalu'},
             },
            {'id': 3, 'annotations': [
                {'id': 95, 'completed_by': 1, 'result': [
                    {"value":
                         {"start": 5, "end": 10, "text": "xanax", "labels": ["B-DRUG"]}, "id": "test1",
                     "from_name": "label", "to_name": "text", "type": "labels", "origin": "prediction"},
                    {'value':

                         {'start': 16, 'end': 23, 'text': 'peavalu', 'labels': ['B-SYMP']}, 'id': 'test2',
                     'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'}],
                 }],

             'data': {
                 'text': 'test xanax test peavalu'},
             }
        ]

        results = PredictionResults(tasks=data, bert_annotations=['B-SYMP', 'B-DRUG'])

        self.assertIs(results.entries[0].choice, True)
        self.assertIs(results.entries[1].choice, False)
        self.assertIs(results.entries[2].choice, None)


if __name__ == '__main__':
    unittest.main()
