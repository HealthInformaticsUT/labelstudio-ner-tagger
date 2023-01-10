from typing import Sequence, Any, List
from estnltk.taggers import Tagger
from estnltk.taggers import RegexTagger

s_c = r"(?P<SPECIALTY_CODE>([EN][0-9]{2,3}))?"
s = r"(?P<SPECIALTY>([-A-ZÜÕÖÄŠŽa-züõöäšž, \(\)]{1,50}))\s*\n"
pattern = s_c + r"\s-\s" + s + r"\s?"

d: Any = {}
d["grammar_symbol"] = "SPECIALTY"
d["regex_type"] = "specialty"
d["_regex_pattern_"] = pattern
d["_group_"] = 0
d["_priority_"] = 0
d["value"] = "whatever"
d["specialty_code"] = lambda m: m.group("SPECIALTY_CODE")
d["specialty"] = lambda m: m.group("SPECIALTY")


vocabulary = [d]


class SpecialtyTagger(Tagger):
    """
    Tags anonym.
    """

    conf_param = ["tagger"]

    def __init__(
        self,
        output_attributes: Sequence = (
            "grammar_symbol",
            "regex_type",
            "value",
            "_priority_",
            "specialty_code",
            "specialty",
        ),
        conflict_resolving_strategy: str = "MAX",
        overlapped: bool = True,
        output_layer: str = "specialty",
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
            output_layer="specialty",
        )

    def _make_layer(self, text, layers, status):
        return self.tagger.make_layer(text=text, layers=layers, status=status)
