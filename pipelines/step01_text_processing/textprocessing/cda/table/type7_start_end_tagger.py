from typing import Sequence, List
from estnltk.taggers import Tagger
from estnltk.taggers import RegexTagger

date_substitutions = {
    "YEAR": r"(?P<YEAR>((19[0-9]{2})|(20[0-9]{2})|([0-9]{2})))",
    "MONTH": r"(?P<MONTH>(0[1-9]|1[0-2]))",
    "DAY": r"(?P<DAY>(0[1-9]|[12][0-9]|3[01]))",
    "HOUR": r"(?P<HOUR>([0-2][0-9]))",
    "MIN": r"(?P<MIN>([0-6][0-9]))",
}
date_regex_with_time = r"({DAY}\.\s*{MONTH}(\.\s*{YEAR}\s*)?\s*({HOUR}:{MIN})?)".format(**date_substitutions)

type7_voc = [
    {
        "grammar_symbol": "START",
        "regex_type": "xx",
        # 1. tables without header can start with analytes specified in regex_pattern, date is optional
        "_regex_pattern_": "(" + date_regex_with_time + ")?\s*(S,P-Na|WBC|S,P-Glükoos)\s*\d",
        "_group_": 0,
        # should have bigger priority (less important) than other taggers type_x_start_end_tagger
        # because tags tables without header
        # this is taken into account in segments_tagger
        "_priority_": 4,
        "_validator_": lambda m: True,
        "value": lambda m: m.group(0).strip("\n"),
    },
    {
        "grammar_symbol": "END",
        "regex_type": "xx",
        "_regex_pattern_": r"(PLT|S,P-CRP|S,P-Kreatiniin|Protrombiini\s*%|S,P-CA|fS,fP-Triglütseriidid|RDW-CV).*?\n",
        "_group_": 0,
        # should have bigger priority (less important) than other taggers (type_x_start_end_tagger)
        # because tags tables without header
        # this is taken into account in segments_tagger
        "_priority_": 5,
        "_validator_": lambda m: True,
        "value": lambda m: m.group(0).strip("\n"),
    },
]


class Type7StartEndTagger(Tagger):
    """
    """

    conf_param = ["tagger"]

    def __init__(
        self,
        output_attributes: Sequence = ("grammar_symbol", "regex_type", "value", "_priority_"),
        conflict_resolving_strategy: str = "MAX",
        overlapped: bool = True,
        output_layer: str = "printout_type7",
    ):
        self.output_attributes = output_attributes
        self.output_layer = output_layer
        self.input_layers: List[str] = []
        self.tagger = RegexTagger(
            vocabulary=type7_voc,
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
