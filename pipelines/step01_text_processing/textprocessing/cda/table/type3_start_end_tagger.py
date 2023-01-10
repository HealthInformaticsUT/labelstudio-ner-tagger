import re
from typing import Sequence, List
from estnltk.taggers import Tagger
from estnltk.taggers import RegexTagger

# from cda_data_cleaning.fact_extraction.measurement_extraction.taggers.robust_date_number_tagger.date_vocabulary
substitutions = {
    "YEAR": r"(?P<YEAR>((19[0-9]{2})|(20[0-9]{2})|([0-9]{2})))",
    "MONTH": r"(?P<MONTH>(0[1-9]|1[0-2]))",
    "DAY": r"(?P<DAY>(0[1-9]|[12][0-9]|3[01]))",
}
date_regex = r"{DAY}\.\s*{MONTH}(\.\s*{YEAR}\s*)?".format(**substitutions)

type3 = [
    # END must come first because, because sometimes the end of the table is also "|Uuring|...|Mõõtühik|",
    # therefore END must be matched first
    {
        "grammar_symbol": "END",
        "regex_type": "xx",
        # 1. two following lines do not contain vertical bar
        # 2. end of the file and contains |
        "_regex_pattern_": "((?!.*\n\|)(?!.*\n.*\n\|).+)|(\|\s*$)",  # "((?!.*\n\|).+)|(\|\s*$)",
        "_group_": 0,
        "_priority_": 1,
        "_validator_": lambda m: bar_validator(m),  # end line must contain a vertical bar
        "value": lambda m: m.group(0).strip("\n"),
    },
    {
        "grammar_symbol": "START",
        "regex_type": "xx",
        # 1. table startwith |Uuring|date1|date2|...|Mõõtühik|
        # 2. if table is without a header then it is in the beginning of the text
        # 3. row contains two "|" and previous lines are emtpy and the one before previous does NOT contain (|)
        #    e.g. the following table will be extracted as one
        #       |0.50|E9/L|
        #       | MONO%| |11.9|8.0| |3.6|3.9|9.2|
        #       |07:10|12:00|
        #       |7.3|7.5|
        #
        #       |6.2|%|
        #       | NEUT#| |4.77|4.17| |10.02|8.75|5.39|
        #       |07:10|12:00|
        #       |6.11|5.71|
        # 4. TODO potentially can tables without header occur in the middle of the text
        #    TODO needs regex that finds rows where previous row does not contain '|'
        "_regex_pattern_": r"(\|(\s*[Uu]uring\s*|<ANONYM.*>)\|("
        + date_regex
        + "\|)+Mõõt(ühik)?\|)"  # 1.
        + "|(^\s*\|.*)|"  # 2.
        + "((?<!.*\|\s*)\n\s*\n\|.*\|)",  # "\n\s*\n\|.*\|",  # 3.
        "_group_": 0,
        "_priority_": 1,
        "_validator_": lambda m: True,
        "value": lambda m: m.group(0).strip("\n"),
    },
]


def bar_validator(m):
    """
    Checks if the row contains a vertical bar.
    """
    if re.match("\|", m.group(0)):
        return True
    return False


class Type3StartEndTagger(Tagger):
    """
    """

    conf_param = ["tagger"]  # , "bar_validator"]

    def __init__(
        self,
        output_attributes: Sequence = ("grammar_symbol", "regex_type", "value", "_priority_"),
        conflict_resolving_strategy: str = "MAX",
        overlapped: bool = True,
        output_layer: str = "printout_type3",
    ):
        self.output_attributes = output_attributes
        self.output_layer = output_layer
        self.input_layers: List[str] = []
        self.tagger = RegexTagger(
            vocabulary=type3,
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
