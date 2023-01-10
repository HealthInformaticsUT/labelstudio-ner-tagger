from typing import List

from .event_date_tagger import EventDateTagger
from .header_word_tagger import HeaderWordTagger
from ..anon.anonym_tagger import AnonymTagger
from .doctor_tagger import DoctorTagger
from .specialty_tagger import SpecialtyTagger
from .doctor_code_tagger import DoctorCodeTagger

from estnltk.taggers import GapTagger
from estnltk.taggers import MergeTagger
from estnltk.taggers import Tagger


class EventTokenTagger(Tagger):
    """
    Tags parts of measurements.
    """

    conf_param = [
        "event_date_tagger",
        "header_word_tagger",
        "anonym_tagger",
        "doctor_tagger",
        "specialty_tagger",
        "doctor_code_tagger",
        "gaps_tagger",
        "merge_tagger",
    ]

    input_layers: List[str] = []

    def __init__(
        self,
        output_attributes=("grammar_symbol", "unit_type", "value", "specialty_code", "specialty", "regex_type"),
        output_layer: str = "grammar_tags",
    ):

        self.output_attributes = output_attributes
        self.output_layer = output_layer

        self.event_date_tagger = EventDateTagger()
        self.header_word_tagger = HeaderWordTagger()
        self.anonym_tagger = AnonymTagger()
        self.doctor_tagger = DoctorTagger()
        self.specialty_tagger = SpecialtyTagger()
        self.doctor_code_tagger = DoctorCodeTagger()

        def gaps_decorator(text: str):
            return {"grammar_symbol": "RANDOM_TEXT"}

        def trim(t: str) -> str:
            t = t.strip()
            t = t.strip("-")
            t = t.strip("=")
            t = t.strip("–")
            t = t.strip(":")
            t = t.strip(".")
            t = t.strip("|")
            t = t.strip("(")
            t = t.strip(")")
            t = t.strip("<")
            t = t.strip(",")
            t = t.strip()
            return t

        self.gaps_tagger = GapTagger(
            output_layer="gaps",
            input_layers=["event_dates", "header_words", "anonym", "doctor", "doctor_code", "specialty"],
            decorator=gaps_decorator,
            trim=trim,
            output_attributes=["grammar_symbol"],
            ambiguous=True,
        )

        self.merge_tagger = MergeTagger(
            output_layer=self.output_layer,
            input_layers=["event_dates", "header_words", "anonym", "doctor", "doctor_code", "specialty", "gaps"],
            output_attributes=self.output_attributes,
        )

    def _make_layer(self, text, layers: dict, status):
        tmp_layers = layers.copy()
        tmp_layers["event_dates"] = self.event_date_tagger.make_layer(text=text, layers=tmp_layers, status=status)
        tmp_layers["header_words"] = self.header_word_tagger.make_layer(text=text, layers=tmp_layers, status=status)
        tmp_layers["anonym"] = self.anonym_tagger.make_layer(text=text, layers=tmp_layers, status=status)
        tmp_layers["doctor"] = self.doctor_tagger.make_layer(text=text, layers=tmp_layers, status=status)
        tmp_layers["specialty"] = self.specialty_tagger.make_layer(text=text, layers=tmp_layers, status=status)
        tmp_layers["doctor_code"] = self.doctor_code_tagger.make_layer(text=text, layers=tmp_layers, status=status)
        tmp_layers["gaps"] = self.gaps_tagger.make_layer(text=text, layers=tmp_layers, status=status)

        return self.merge_tagger.make_layer(text=text, layers=tmp_layers, status=status)
