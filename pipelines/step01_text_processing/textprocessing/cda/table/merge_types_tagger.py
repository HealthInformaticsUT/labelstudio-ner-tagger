from typing import List

from estnltk.taggers import Tagger

from .type1_start_end_tagger import Type1StartEndTagger
from .type2_start_end_tagger import Type2StartEndTagger
from .type3_start_end_tagger import Type3StartEndTagger
from .type4_start_end_tagger import Type4StartEndTagger
from .type5_start_end_tagger import Type5StartEndTagger
from .type6_start_end_tagger import Type6StartEndTagger

from estnltk.taggers import MergeTagger, FlattenTagger


class MergeTypesTagger(Tagger):
    """

    """

    conf_param = [
        "type1_start_end_tagger",
        "type2_start_end_tagger",
        "type3_start_end_tagger",
        "type4_start_end_tagger",
        "type5_start_end_tagger",
        "type6_start_end_tagger",
        "merge_tagger",
    ]

    def __init__(
        self,
        output_attributes=("grammar_symbol", "value"),
        conflict_resolving_strategy: str = "MAX",
        overlapped: bool = True,
        output_layer: str = "grammar_tags",
    ):
        self.input_layers: List[str] = []  # ['words']
        self.output_attributes = tuple(output_attributes)
        self.output_layer = output_layer
        # priority_attribute = '_priority_'

        self.type1_start_end_tagger = Type1StartEndTagger()
        self.type2_start_end_tagger = Type2StartEndTagger()
        self.type3_start_end_tagger = Type3StartEndTagger()
        self.type4_start_end_tagger = Type4StartEndTagger()
        self.type5_start_end_tagger = Type5StartEndTagger()
        self.type6_start_end_tagger = Type6StartEndTagger()

        # self.flatten_tagger = FlattenTagger(
        #    input_layer="drug_phrase",
        #    output_layer="drug_phrase_flat",
        #    output_attributes=["grammar_symbol", "value", "_priority_"],
        # )

        self.merge_tagger = MergeTagger(
            output_layer="grammar_tags",
            input_layers=["type1", "type2", "type3", "type4", "type5", "type6"],
            output_attributes=["grammar_symbol", "value"],
        )

    def _make_layer(self, text, layers, status):
        tmp_layers = {}
        tmp_layers["type1"] = self.type1_start_end_tagger.make_layer(text=text, layers=tmp_layers, status=status)
        tmp_layers["type2"] = self.type2_start_end_tagger.make_layer(text=text, layers=tmp_layers, status=status)
        tmp_layers["type3"] = self.type3_start_end_tagger.make_layer(text=text, layers=tmp_layers, status=status)
        tmp_layers["type4"] = self.type4_start_end_tagger.make_layer(text=text, layers=tmp_layers, status=status)
        tmp_layers["type5"] = self.type5_start_end_tagger.make_layer(text=text, layers=tmp_layers, status=status)
        tmp_layers["type6"] = self.type6_start_end_tagger.make_layer(text=text, layers=tmp_layers, status=status)

        # merge layer võiks olla atribuut ambigous
        tmp_layers["type1"].ambiguous = True

        print(self.merge_tagger)
        return self.merge_tagger.make_layer(text=text, layers=tmp_layers, status=status)
