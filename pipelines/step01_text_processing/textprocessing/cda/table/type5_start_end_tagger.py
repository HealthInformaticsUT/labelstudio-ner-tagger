from typing import Sequence, List
from estnltk.taggers import Tagger
from estnltk.taggers import RegexTagger

substitutions = {
    "YEAR": r"(?P<YEAR>((19[0-9]{2})|(20[0-9]{2})|([0-9]{2})))",
    "MONTH": r"(?P<MONTH>(0[1-9]|1[0-2]))",
    "DAY": r"(?P<DAY>(0[1-9]|[12][0-9]|3[01]))",
    "HOUR": r"(?P<HOUR>([0-2][0-9]))",
    "MIN": r"(?P<MIN>([0-5][0-9]))",
    "SEC": r"(?P<MIN>([0-5][0-9]))",
}
date_regex = r"{DAY}\.\s*{MONTH}(\.\s*{YEAR}\s*)?(\s*{HOUR}:{MIN}:{SEC})?|{YEAR}\.\s*{MONTH}\.\s*{DAY}".format(
    **substitutions
)

analysis_substitutions = {
    "AN_NAMES": "(Biokeemia analüüs|Kliinili\s*ne vere\s*(ja glükohemo\s*globiini\s*)?analüüs|Hemostasiogramm|"
    "Kl keemia ja immunoloogia automaatliinil teostatud uuring|Allergia testid|Happe-alus tasakaal|"
    "Immuunmeetoditel\s*uuringud|Uriini analüüs|Mikrobioloogia analüüs|Uriini analüüs)"
}
analysis_name_regex = r"({AN_NAMES})\s*".format(**analysis_substitutions)

type5_voc = [
    {
        "grammar_symbol": "START",
        "regex_type": "xx",
        # 1.1 header contains "Analüüs Tellitud Ühik Referents" followed one or more dates
        # 1.2 header contains "Analüüs/parameeter Referentsväärtused" followed one or more dates
        # 1.3 header contains "Analüüsi nimetus Parameeter Referents" followed one or more dates
        # 2. header contain analysis name followed by "Rerentsväärtus" and one or more dates
        #    (can contain hours and minutes)
        "_regex_pattern_": r"((Analüüsid\s*)?_?Analü\s*ü\s*s\s*(\\\s*Tellitud\s*)?_?Ühi\s*k_?\s*Ref\s*er\s*ents_?\s*("  # 1.1
        + date_regex  # 1.1
        + "\s*)+)|"  # 1.1
        + r"(Analüüs/parameeter Referentsväärtused\s*("  # 1.2
        + date_regex  # 1.2
        + "\s*)+)|"  # 1.2
        + r"(Anal\s*üüsi nimetus\s*Parameeter\s*Referents\s*("  # 1.3
        + date_regex  # 1.3
        + "\s*)+)|"  # 1.2
        + "("
        + analysis_name_regex
        + "(Test)?"  # 2.
        + "\s*Re(fe)?r\s*entsvää\s*rtus\s*("  # 2.
        + date_regex  # 2.
        + ".*?\s*)+(\s*Materjal)?)",  # 2.
        "_group_": 0,
        "_priority_": 1,
        "_validator_": lambda m: True,
        "value": lambda m: m.group(0).strip("\n"),
    },
    {
        "grammar_symbol": "END",
        "regex_type": "xx",
        # matches a line that is followed by
        # 0. two empty lines
        # 1. line break + certain medical pharses
        # 1.2 certain medical pharses (some texts are written without line breaks)
        # 2. line break + analysis name
        # 3. some texts are written without any \n
        # 4. a new table is starting with date, do not include the date,
        #    instead include the last three letter before the date
        "_regex_pattern_": "\n\s?\n\s?\n|"
        # mainly for start 1.
        + "\n(.*)(?="
        + "Raviarst|"  # 1.
        + "Re[ž_]iim(i)? ja ravialased soovitused|"  # 1.
        + "[Rr]avimid|"  # 1.
        + "Operatisoon|"  # 1.
        + "Kokkuvõte patsiendi ravist|"  # 1.
        + "Retsept|"  # 1.
        # mainly for start 2.
        + analysis_name_regex
        + ")|"  # 2.
        + "(Re[ž_]iim ja ravialased soovitused)|"  # 1.2
        + "(Retseptid:)|"  # 3.
        + "(Eluanamnees)|"
        + "\S{3,}\s*(?="  # 4.
        + date_regex  # 4.
        + ")",  # 4.
        "_group_": 0,
        "_priority_": 1,
        "_validator_": lambda m: True,
        "value": lambda m: m.group(0).strip("\n"),
    },
]


class Type5StartEndTagger(Tagger):
    """

    """

    conf_param = ["tagger"]

    def __init__(
        self,
        output_attributes: Sequence = ("grammar_symbol", "regex_type", "value", "_priority_"),
        conflict_resolving_strategy: str = "MAX",
        overlapped: bool = True,
        output_layer: str = "printout_type5",
    ):
        self.output_attributes = output_attributes
        self.output_layer = output_layer
        self.input_layers: List[str] = []
        self.tagger = RegexTagger(
            vocabulary=type5_voc,
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
