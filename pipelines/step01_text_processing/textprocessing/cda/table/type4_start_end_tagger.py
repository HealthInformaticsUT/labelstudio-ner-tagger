from typing import Sequence, List
from estnltk.taggers import Tagger
from estnltk.taggers import RegexTagger

substitutions = {
    "YEAR": r"(?P<YEAR>((19[0-9]{2})|(20[0-9]{2})|([0-9]{2})))",
    "MONTH": r"(?P<MONTH>(0[1-9]|1[0-2]))",
    "DAY": r"(?P<DAY>(0[1-9]|[12][0-9]|3[01]))",
}
date_regex = r"{DAY}\.\s*{MONTH}(\.\s*{YEAR}\s*)?".format(**substitutions)

# e.g. epi_id = 120541689, id = 943140, table=anamnesis, column = anamnesis
type4_voc = [
    {
        "grammar_symbol": "START",
        "regex_type": "xx",
        "_regex_pattern_": "(((Proovi)?\s*v\s*õtmise\s*kuupäev|Tellimise\s*kuupäev)\s*Teostamise kuupä\s*ev\s*.*?\s*Lühend\s*Tulemus\s*Ühik\s*Referents\s*HK\s*ko\s*o\s*d\s*)|"
        + "(Nä\s*ita\s*Refe\s*rentsväärt\s*usi\s*ja\s*ühikuid\s*Analüüsid\s*)|"
        + "(Analüüsid\s*Kliiniline vere ja glükoh\s*emoglobiini analüüs\s*Rerentsväärtus\s*"
        + date_regex
        + ")|"
        + "(Analüüs\s*Parameeter\s*Tu\s*lemus\s*Refe\s*rentsvää\s*rtus\s*(/|,)\s*ot\s*sustuspiirid ja ühik)|"
        + "(Analüüs\s*Parameeter\s*Referents\s*vää\s*rtus\s*((/|,) otsustuspiir(id)?)?\s*ja\s*üh\s*ik)|"
        + "(Analüüsi nimetus P\s*arameeter Referents)|"
        + "(Analüüs\s*Parameeter\s*Ühik\s*Referentsväärtus\s*Tulemused\s*)|"
        + "(Analüüsid\s*Nimetus\s*Referentsväärtus\s*Tulemused\s*)|"
        + "(Analüüs\s*Parameeter\s*Vastus)",
        "_group_": 0,
        "_priority_": 1,
        "_validator_": lambda m: True,
        "value": lambda m: m.group(0).strip("\n"),
    },
    {
        "grammar_symbol": "END",
        "regex_type": "xx",
        # if the following line starts with a "Väljastatud dokumendid" or "Töövõimetusleht" or "Raviarst/koostaja",
        # or "Teostaja" or "Kokkuvõte patsiendi ravist" or "Proovi võtmise kuupäev" then table is over
        # IDEA ! 2. next line does not contain a number (analysis table have values as numbers)
        "_regex_pattern_": "\n(.*)\n(.*)\n(?="
        + "\s*(Väljastatu\s*d\s*dok\s*umendid|"
        + "töövõimetusleht|"
        + "tavi\s*a\s*rst\s*/ko\s*osta\s*ja|"
        + "teosta\s*ja|"
        + "Kokku\s*võte patsiendi ra\s*vist|"
        + "Proovi võtmise kuupäev|"
        + "laboriuuringute vastus|"
        + "märkused|"
        + "kokkuvõte|"
        + "ravi|"
        + "diagnoos|"
        + "uuringud)"
        + ")",
        "_group_": 0,
        "_priority_": 1,
        "_validator_": lambda m: True,
        "value": lambda m: m.group(0).strip("\n"),
    },
]


class Type4StartEndTagger(Tagger):
    """

    """

    conf_param = ["tagger"]

    def __init__(
        self,
        output_attributes: Sequence = ("grammar_symbol", "regex_type", "value", "_priority_"),
        conflict_resolving_strategy: str = "MAX",
        overlapped: bool = True,
        output_layer: str = "printout_type4",
    ):
        self.output_attributes = output_attributes
        self.output_layer = output_layer
        self.input_layers: List[str] = []
        self.tagger = RegexTagger(
            vocabulary=type4_voc,
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
