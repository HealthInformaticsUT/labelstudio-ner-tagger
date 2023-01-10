from typing import Tuple
from typing import MutableMapping
from estnltk import ElementaryBaseSpan
from estnltk.layer.layer import Layer
from estnltk.text import Text
from estnltk.taggers import Tagger


class SegmentsTaggerOneType(Tagger):
    """
    """

    conf_param = ["tagger"]

    def __init__(
        self,
        input_layer: str = "printout_type1",  # needs to be specified when creating the tagger, e.g. "type2"
        output_layer: str = "type1_segments",  # needs to be sepcified when creating the tagger, e.g. "type2_segments"
        output_attributes: Tuple[str, str, str, str] = ("grammar_symbol", "regex_type", "value", "_priority_"),
    ):
        self.input_layers = [input_layer]

        self.output_attributes = output_attributes
        if output_attributes is None:
            self.output_attributes = ()
        self.output_layer = output_layer

    def _make_layer(self, text: Text, layers: MutableMapping[str, Layer], status: dict) -> Layer:
        headers_layer = layers[self.input_layers[0]]

        layer = Layer(self.output_layer, attributes=self.output_attributes, text_object=text)

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
                layer.add_annotation(
                    ElementaryBaseSpan(span_start, span_end),
                    value=text.text[span_start:span_end],
                    _priority_=span._priority_,
                    ambiguous=True,
                    regex_type=[self.input_layers][0],
                )
                span_start = None  # need to find a new "START" in the next iteration

        # if no annotation added, return an empty layer
        if not layer.spans:
            print("No annotations :(")
            # should be removed in the future
            layer.add_annotation(ElementaryBaseSpan(0, 0))

        return layer
