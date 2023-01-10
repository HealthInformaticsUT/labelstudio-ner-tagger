import re
from typing import Sequence, List
from estnltk.taggers import Tagger
from estnltk.taggers import RegexTagger

date_substitutions = {
    "YEAR": r"(?P<YEAR>((19[0-9]{2})|(20[0-9]{2})|([0-9]{2})))",
    "MONTH": r"(?P<MONTH>(0[1-9]|1[0-2]))",
    "DAY": r"(?P<DAY>(0[1-9]|[12][0-9]|3[01]))",
    "HOUR": r"(?P<HOUR>([0-2][0-9]))",
    "MIN": r"(?P<MIN>([0-5][0-9]))",
}
date_regex = r"({DAY}\.\s*{MONTH}(\.\s*{YEAR}\s*)?)".format(**date_substitutions)
datetime_regex = r"({DAY}\.\s*{MONTH}(\.\s*{YEAR}\s*)?(\s*{HOUR}:{MIN}))?".format(**date_substitutions)

# ignore_case = True
analysis_substitutions = {
    "AN_NAMES": r"(hemogramm|"
    + r"uriini\s*ribaanalüüs|uriini\s*analüüs\s*testribaga|uriini\s*ribatest|"
    + r"vereäige\s*mikroskoopiline\s*uuring|kliiniline\s*veri|vere\s*biokeemia|uriini\s*sademe\s*mikroskoopia|"
    + r"hematoloogilised\s*ja\s*uriini\s*uuringud|klii\s*nilise\s*keemia\s*u\s*uringud|"
    + r"hematoloogilised\s*uuringud|immuunmeetoditel\s*põhinevad\s*uuringud|Hematoloogia\s*labori\s*analüüsid|"
    + r"biokeemilised\s*uuringud|MP\s*Hormoonid,\s*kasvajamarkerid\s*jm\.\s*immuunuuringud\s*|vereäige mikroskoopia|"
    + r"seroloogilised uuringud MB-s|lipiidid ja südamemarkerid|diabeedi uuringud|aneemia uuringud|aneemiamarkerid ja vitamiinid|"
    + r"hormoon- jm\. immuunuuringud|(auto)?immuunuuringud|nakkushaiguste diagnostika|immuunsüsteemi uuringud|"
    + r"hüübimissüsteemi uuringud\s*INR|haigustekitajate serodiagnostika)\s*",
    "AN": r"(analüüsid)\s*",
}
analysis_name_regex = r"({AN_NAMES})\s*".format(**analysis_substitutions)
analysis_regex = r"({AN}|{AN_NAMES})\s*".format(**analysis_substitutions)

type6_voc = [
    {
        "grammar_symbol": "START",
        "regex_type": "xx",
        # 1. date first, then name
        # 2. name first, then date
        # 3. only names
        "_regex_pattern_": analysis_regex
        + date_regex
        + "|"  # 1.
        + date_regex
        + analysis_regex
        + "|"  # 2.
        + analysis_name_regex
        + ".*",  # 3.
        "_group_": 0,
        # should have bigger priority (less important) than other taggers type_x_start_end_tagger except type7
        # this is taken into account in segments_tagger
        "_priority_": 2,
        "_validator_": lambda m: True,
        "value": lambda m: m.group(0).strip("\n"),
    },
    {
        "grammar_symbol": "END",
        "regex_type": "xx",
        # 1. matches a line that is followed by two empty lines
        # 2. end of the file (ending parenthesis or ] or word)
        # 3. following line starts with
        #   3.0 "LEID"
        #   3.1 "Ravi:"
        #   3.2 "Diagnoos:"
        #   3.3 "KAEBUSED, KÄESOLEVAD PROBLEEMID"
        # 5. current row contains ] and following row does not contain ] and date
        #       1. does match as end     e.g 12.03.2019 10:23: HbA1c % 7,0 % [norm 4 - 6]
        #           Hemogramm
        #       2. does not match as end e.g 25.10.2019 12:05:17: Kreatiniin
        # 6. two empty rows followed by a date
        # 7. if next line starts with optional date and another analysis name (sometimes starts with *)
        "_regex_pattern_": r"(.\n\n\n)|"  # 1.
        + r"((\)|\]|\w*)\s*$)|"  # 2.  # r"((\w*|.*?\))\s*$)|"
        + r"(\n(.*)(?=\s*LEID))|"  # 3.0
        + r"(\n(.*)(?=\s*Ravi:))|"  # 3.1.
        + r"(\n(.*)(?=\s*Diagnoos:))|"  # 3.2.
        + r"\n(.*)(?=KAEBUSED, KÄESOLEV PROBLEEM)|"  # 3.3
        + r"(\]\n(?!.*\])(?!.*?\d{2}\.\d{2}\.\d{4}))|"  # 5.
        # works well, but super slow
        # + r"(.*(?=\n\s*\n"  # 6.
        # + "\d{2}\.\d{2}\.\d{4}"#date_regex  # 6.
        # + "))|"  # 6.
        + r"\n(.*)(?=\S+" + datetime_regex + r"?\s*\\*?{AN_NAMES})".format(**analysis_substitutions),  # 7.
        "_group_": 0,
        # should have bigger priority (less important) than other taggers (type_x_start_end_tagger) except type7
        # because it is not as accurate as others
        # this is taken into account in segments_tagger
        "_priority_": 3,
        "_validator_": lambda m: True,
        "value": lambda m: m.group(0).strip("\n"),
    },
]


class Type6StartEndTagger(Tagger):
    """
    """

    conf_param = ["tagger"]

    def __init__(
        self,
        output_attributes: Sequence = ("grammar_symbol", "regex_type", "value", "_priority_"),
        conflict_resolving_strategy: str = "MAX",
        overlapped: bool = True,
        output_layer: str = "printout_type6",
    ):
        self.output_attributes = output_attributes
        self.output_layer = output_layer
        self.input_layers: List[str] = []
        self.tagger = RegexTagger(
            vocabulary=type6_voc,
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
