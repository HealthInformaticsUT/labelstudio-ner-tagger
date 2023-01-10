from typing import MutableMapping
from estnltk import Tagger, Text, Layer
from estnltk.layer.base_span import ElementaryBaseSpan


def decorator(span):
    return {
        "header": span.enclosing_text.strip(),
        "header_offset": span.start,
        **span.annotations[0],
    }


def validator(span):
    assert len(span.annotations) == 1
    return True


class EventSegmentsTagger(Tagger):
    """Segments text by events layer.

    """

    conf_param = ["decorator", "validator", "include_header"]

    def __init__(self):
        self.output_attributes = [
            "header",
            "header_offset",
            "DATE",
            "HEADERWORD",
            "DOCTOR",
            "DOCTOR_CODE",
            "SPECIALTY",
            "SPECIALTY_CODE",
            "ANONYM",
        ]
        self.input_layers = ["event_headers"]  # output layer of the EventHeaderTagger
        self.output_layer = "event_segments"
        self.decorator = decorator
        self.validator = validator
        self.include_header = False

    def _make_layer(self, text: Text, layers: MutableMapping[str, Layer], status: dict) -> Layer:
        headers_layer = layers[self.input_layers[0]]

        layer = Layer(self.output_layer, text_object=text, attributes=self.output_attributes)#, ambiguous = True)

        if self.include_header:
            last_span = None
            for span in headers_layer:
                if not self.validator(span):
                    continue

                if last_span is None:
                    if span.start > 0:
                        layer.add_annotation(ElementaryBaseSpan(0, span.start))
                else:
                    layer.add_annotation(ElementaryBaseSpan(last_span.start, span.start), **self.decorator(last_span))
                last_span = span

            if last_span is None:
                layer.add_annotation(ElementaryBaseSpan(0, len(text.text)))
            else:
                layer.add_annotation(ElementaryBaseSpan(last_span.start, len(text.text)), **self.decorator(last_span))
        else:
            last_span = None
            for span in headers_layer:
                if not self.validator(span):
                    continue

                if last_span is None:
                    if span.start > 0:
                        layer.add_annotation(ElementaryBaseSpan(0, span.start))
                else:
                    # TODO: remove hack max(last_span.end, span.start), replace with span.start
                    #  first make sure that headers_layer has no conflicts
                    layer.add_annotation(
                        ElementaryBaseSpan(last_span.end, max(last_span.end, span.start)), **self.decorator(last_span)
                    )
                last_span = span

            if last_span is None:
                layer.add_annotation(ElementaryBaseSpan(0, len(text.text)))
            else:
                layer.add_annotation(ElementaryBaseSpan(last_span.end, len(text.text)), **self.decorator(last_span))

        return layer
