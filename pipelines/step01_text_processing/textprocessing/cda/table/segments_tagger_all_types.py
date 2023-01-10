from typing import Tuple, List
from typing import MutableMapping
from estnltk import ElementaryBaseSpan
from estnltk.layer.layer import Layer
from estnltk.layer_operations import resolve_conflicts
from estnltk.text import Text
from estnltk.taggers import Tagger


class SegmentsTagger(Tagger):
    """
    """

    conf_param = ["tagger"]

    def __init__(
        self,
        input_layer: List[str] = [
            "printout_type1",
            "printout_type2",
            "printout_type3",
            "printout_type4",
            "printout_type5",
            "printout_type6",
            "printout_type7",
        ],
        output_layer: str = "printout_segments",
        output_attributes: Tuple[str, str, str, str] = ("grammar_symbol", "regex_type", "value", "_priority_"),
    ):
        self.input_layers = input_layer

        self.output_attributes = output_attributes
        if output_attributes is None:
            self.output_attributes = ()
        self.output_layer = output_layer
        # self.conflict_resolving_strategy = 'MAX'
        self.tagger = None

    def _make_layer(self, text: Text, layers: MutableMapping[str, Layer], status: dict) -> Layer:
        layer = Layer(self.output_layer, attributes=self.output_attributes, text_object=text)

        for i, input_layer in enumerate(self.input_layers):
            headers_layer = layers[self.input_layers[i]]

            span_start = None

            # spans starts from the first grammar_symbol "start"
            # span ends from the FIRST grammar_symbol "end" AFTER "start"
            for span in headers_layer:

                # search for a start, if there are multiple ones, the one right before "END" is chosen
                if span.grammar_symbol == "START":
                    span_start = span.start
                # search for a following end, when found, then we can add an annotation
                if span.grammar_symbol == "END" and span_start is not None:
                    span_end = span.end
                    # type 6 is less prioritized than others
                    layer.add_annotation(
                        ElementaryBaseSpan(span_start, span_end),
                        value=text.text[span_start:span_end],
                        _priority_=span._priority_,
                        ambiguous=True,
                        regex_type=input_layer,
                    )
                    span_start = None  # need to find a new "START" in the next iteration

            # if no annotation added, return an empty layer
            if not layer.spans:
                # should be removed in the future
                layer.add_annotation(ElementaryBaseSpan(0, 0), ambiguous=True)
            layer.ambiguous = True

        # some types are overlapping, in order to not get one text multiple times, keep the longer version of the text
        resolve_conflicts(layer, conflict_resolving_strategy="MAX", priority_attribute="_priority_")

        return layer
