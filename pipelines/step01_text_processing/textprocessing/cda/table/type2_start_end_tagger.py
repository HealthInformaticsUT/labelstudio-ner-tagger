from typing import Sequence, List
from estnltk.taggers import Tagger
from estnltk.taggers import RegexTagger

date_regex = r"((0[1-9]|[12][0-9]|3[01])\.(0[1-9]|1[012])\.(20[0-9]{2})){1}"  # e.g. 12.03.2020

substitutions = {
    "AN": r"(ANALÜÜSI(DE)?\s*TELLIMUS\s*nr:\s*\d*\s*)",
    "MAT": r"(MATERJAL(ID)?:\s*.*?\s*.*?\s*)",  # "MAT": r"(MATERJAL:\s*(.*?\s*){0,4})\s*",
    "VAS": r"(VASTUSED:.*?)",
    "MARK": r"(Märkus:(.*?\s*){0,2}\s*)",
}

type2 = [
    {
        "grammar_symbol": "START",
        "regex_type": "xx",
        "_regex_pattern_": r"({AN}{MAT}({MARK})?{VAS}|{AN}{MARK}?{VAS}|{AN}{MAT}|{MAT}{VAS}|{VAS}|{AN})".format(
            **substitutions
        ),
        "_group_": 0,
        "_priority_": 0,  # must have smaller priority (more important) than type1
        "_validator_": lambda m: True,
        "value": lambda m: m.group(0).strip("\n"),
    },
    {
        "grammar_symbol": "END",
        "regex_type": "xx",
        # 1. matches if there is a date (new analysis starts)
        # 2. matches the end of a file
        # 3. matches if followed by [1] Väline teostaja
        # "(?:(?!X).)*" = "search until X but not including X"
        "_regex_pattern_": r"\n(?="  # "(\n"
        + date_regex
        + r")|"
        + r"((\)|\w+)\s*$)|"  # \n(.*)\n(.*)\n(?=" + date_regex + ")", "((\w*|.*?\))\s*$)"
        + r"((.*\n)(?=\[1\]\s*Väline teostaja))",
        "_group_": 0,
        "_priority_": 0,  # must have smaller priority (more important) than type1
        "_validator_": lambda m: True,
        "value": lambda m: m.group(0).strip("\n"),
    },
]


class Type2StartEndTagger(Tagger):
    """
    """

    conf_param = ["tagger"]

    def __init__(
        self,
        output_attributes: Sequence = ("grammar_symbol", "regex_type", "value", "_priority_"),
        conflict_resolving_strategy: str = "MAX",
        overlapped: bool = True,
        output_layer: str = "printout_type2",
    ):
        self.output_attributes = output_attributes
        self.output_layer = output_layer
        self.input_layers: List[str] = []
        self.tagger = RegexTagger(
            vocabulary=type2,
            output_attributes=output_attributes,
            conflict_resolving_strategy=conflict_resolving_strategy,
            priority_attribute="_priority_",
            overlapped=overlapped,
            ambiguous=False,
            ignore_case=False,  # needs to only match VASTUSED and not vastused
            output_layer=output_layer,
        )

    def _make_layer(self, text, layers, status):
        return self.tagger.make_layer(text=text, layers=layers, status=status)
