from typing import Sequence, List
from estnltk.taggers import Tagger
from estnltk.taggers import RegexTagger

substitutions = {
    "YEAR": r"(?P<YEAR>((19[0-9]{2})|(20[0-9]{2})|([0-9]{2})))",
    "MONTH": r"(?P<MONTH>(0[1-9]|1[0-2]))",
    "DAY": r"(?P<DAY>(0[1-9]|[12][0-9]|3[01]))",
}
date_regex = r"{DAY}\.\s*{MONTH}(\.\s*{YEAR}\s*)?".format(**substitutions)

type1_voc = [
    {
        "grammar_symbol": "START",
        "regex_type": "xx",
        "_regex_pattern_": date_regex
        + "\s*\d*\s*(line veri\s*)?(Kliiniline veri|Uriin)\s*",  # date_regex + ".*?\s*(Kliiniline veri|Uriin)\s*",
        "_group_": 0,
        "_priority_": 1,
        "_validator_": lambda m: True,
        "value": lambda m: m.group(0).strip("\n"),
    },
    {
        "grammar_symbol": "END",
        "regex_type": "xx",
        # kliiniline veri is followed by RBC and Uriin by Erikaal
        "_regex_pattern_": "(RBC.*;|Erikaal:.*;)",  # date_regex + '\s*Kliiniline veri\s*' +  '(.*?)(.*?)\n',
        "_group_": 0,
        "_priority_": 1,
        "_validator_": lambda m: True,
        "value": lambda m: m.group(0).strip("\n"),
    },
]


class Type1StartEndTagger(Tagger):
    """

    """

    conf_param = ["tagger"]

    def __init__(
        self,
        output_attributes: Sequence = ("grammar_symbol", "regex_type", "value", "_priority_"),
        conflict_resolving_strategy: str = "MAX",
        overlapped: bool = True,
        output_layer: str = "printout_type1",
    ):
        self.output_attributes = output_attributes
        self.output_layer = output_layer
        self.input_layers: List[str] = []
        self.tagger = RegexTagger(
            vocabulary=type1_voc,
            output_attributes=output_attributes,
            conflict_resolving_strategy=conflict_resolving_strategy,
            priority_attribute="_priority_",
            overlapped=overlapped,
            ambiguous=False,
            ignore_case=True,
            output_layer=output_layer,
        )

    def _make_layer(self, text, layers, status):
        return self.tagger.make_layer(text=text, layers=layers, status=status)
