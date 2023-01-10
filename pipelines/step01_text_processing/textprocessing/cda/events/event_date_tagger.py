from typing import Sequence, List, Any
from estnltk.taggers import Tagger
from estnltk.taggers import RegexTagger
from .date_regex import regexes

voc: List = []
for key, regex in regexes.items():
    d: Any = {}
    d["grammar_symbol"] = "DATE"
    d["regex_type"] = "date"
    d["_regex_pattern_"] = regex.pattern
    d["_group_"] = 0
    d["_priority_"] = 0
    d["value"] = lambda m: m.group(0).strip()
    voc.append(d)


class EventDateTagger(Tagger):
    """
    Tags event headers.
    """

    conf_param = ["tagger"]

    def __init__(
        self,
        output_attributes: Sequence = ("grammar_symbol", "regex_type", "value", "_priority_"),
        conflict_resolving_strategy: str = "MAX",
        overlapped: bool = True,
        output_layer: str = "event_dates",
    ):
        self.output_attributes = output_attributes
        self.output_layer = output_layer
        self.input_layers: List = []
        self.tagger = RegexTagger(
            vocabulary=voc,
            output_attributes=output_attributes,
            conflict_resolving_strategy=conflict_resolving_strategy,
            priority_attribute="_priority_",
            overlapped=overlapped,
            ambiguous=True,
            output_layer="event_dates",
        )

    def _make_layer(self, text, layers, status):
        return self.tagger.make_layer(text=text, layers=layers, status=status)
