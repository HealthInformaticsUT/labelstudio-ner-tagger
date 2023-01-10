from typing import Sequence, Any, List
from estnltk.taggers import Tagger
from estnltk.taggers import RegexTagger

d: Any = {}
d["grammar_symbol"] = "DOCTOR_CODE"
d["regex_type"] = "doctor_code"
d["_regex_pattern_"] = r"(?P<CODE>([DMKN][0-9]{4,5}))"
d["_group_"] = 0
d["_priority_"] = 0
d["value"] = lambda m: m.group("CODE")

vocabulary = [d]


class DoctorCodeTagger(Tagger):
    """
    Tags anonym.
    """

    conf_param = ["tagger"]

    def __init__(
        self,
        output_attributes: Sequence = ("grammar_symbol", "regex_type", "value", "_priority_"),
        conflict_resolving_strategy: str = "MAX",
        overlapped: bool = True,
        output_layer: str = "doctor_code",
    ):
        self.output_attributes = output_attributes
        self.output_layer = output_layer
        self.input_layers: List = []
        self.tagger = RegexTagger(
            vocabulary=vocabulary,
            output_attributes=output_attributes,
            conflict_resolving_strategy=conflict_resolving_strategy,
            priority_attribute="_priority_",
            overlapped=overlapped,
            ambiguous=True,
            ignore_case=True,
            output_layer="doctor_code",
        )

    def _make_layer(self, text, layers, status):
        return self.tagger.make_layer(text=text, layers=layers, status=status)
