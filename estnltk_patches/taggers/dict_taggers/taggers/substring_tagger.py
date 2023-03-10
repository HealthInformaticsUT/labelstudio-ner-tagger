from copy import copy
from ahocorasick import Automaton

from estnltk import Text, Layer, Tagger
from estnltk import ElementaryBaseSpan, Span, Annotation
from typing import Tuple, List, Dict, Sequence, Union, Any, Callable, Iterator, Generator

from ..extraction_rules.ruleset import Ruleset


class SubstringTagger(Tagger):
    """
    Tags occurrences of substrings on the text, solves possible conflicts and creates a new layer of the matches.
    Uses Aho-Corasick algorithm to efficiently match many substrings simultaneously.

    Static extraction rules are in the form string --> dict where the dict contains the annotation for the match, e.g.
        Washington    --> {type: capital, country: US},
        Tartu         --> {type: town, country: Estonia}
        St. Mary Mead --> {type: village, country: UK}

    Dynamic extraction rules are in the form string --> function where the function recomputes annotations.
    The function takes in the match as Span and must output a dictionary specifying a new annotation or None.
    The output None signals that extracted match is not valid and should be dropped.

    For convenience each extraction rule can be split into static and dynamic parts where the static rule determines
    non-variable annotations and dynamic rule is used to re-compute values of context sensitive attributes.
    The dynamic part itself can be split into, two as well.
    First a global decorator that is applied for all matches and then pattern specific decorators.
    Decorators can return None to filter out unsuitable matches.

    Annotations are added to extracted matches based on the right-hand-side of the matching rules:
    * First statical rules are applied to specify fixed attributes. No spans are dropped!
    * Next the global decorator is applied to update the annotation.
    * A span is dropped when the resulting annotation is not a dictionary of attribute values.
    * Finally decorators from dynamical rules are applied to update the annotation.
    * A span is dropped when the resulting annotation is not a dictionary of attribute values.

    Each rule may additionally have priority and group attributes.
    These can used in custom conflict resolver to choose between overlapping spans.

    Rules can be specified during the initialisation and cannot be changed afterwards.
    Use the class Ruleset and its methods to load static rules from csv files before initialising the tagger.

    TODO: Harden attribute access
    """

    # noinspection PyMissingConstructor,PyUnresolvedReferences
    def __init__(self,
                 ruleset: Ruleset,
                 token_separators: str = '',
                 output_layer: str = 'terms',
                 output_attributes: Sequence = None,
                 global_decorator: Callable[[Text, Span], Union[Dict[str, Any], None]] = None,
                 conflict_resolver: Union[str, Callable[[Layer], Layer]] = 'KEEP_MAXIMAL',
                 ignore_case: bool = False
                 ):
        """
        Initialize a new SubstringTagger instance.

        Parameters
        ----------
        ruleset:
            A list of substrings together with their annotations.
            Can be imported before creation of a tagger from CSV file with Ruleset.load method.
        token_separators:
            A list of characters that determine the end of token.
            If token separators are unspecified then all substring matches are considered.
            Otherwise, a match is ignored if it ends or starts inside of a token.
            To mach a multi-token string the extraction pattern must contain correct separator.
        output_layer:
            The name of the new layer (default: 'terms').
        global_decorator:
            A global decorator that is applied for all matches and can update attribute values or invalidate match.
            It must take in two arguments:
            * text: a text object that is processed
            * span: a span for which the attributes are recomputed.
            The function should return a dictionary of updated attribute values or None to invalidate the match.
        conflict_resolver: 'KEEP_ALL', 'KEEP_MAXIMAL', 'KEEP_MINIMAL' (default: 'KEEP_MAXIMAL')
            Strategy to choose between overlapping matches.
            Specify your own layer assembler if none of the predefined strategies does not work.
            A custom function must be take in two arguments:
            * layer: a layer to which spans must be added
            * triples: a list of (span, group, priority) triples
            and must output the updated layer which hopefully containing some spans.
            These triples can come in canonical order which means:
                span[i].start <= span[i+1].start
                span[i].start == span[i+1].start ==> span[i].end < span[i + 1].end
        ignore_case:
            If True, then matches do not depend on capitalisation of letters
            If False, then capitalisation of letters is important

        Extraction rules are in the form string --> dict where the dict contains the annotation for the match, e.g.
            Washington    --> {type: capital, country: US},
            Tartu         --> {type: town, country: Estonia}
            St. Mary Mead --> {type: village, country: UK}
        Each rule may additionally have priority and group attributes.
        These can used in custom conflict resolver to choose between overlapping spans.

        Token separators define token boundaries. If they are set then the pattern must match the entire token, e.g.
        separators = '|' implies that the second match in '|Washington| Tartu |St. Mary Mead' is dropped.
        The last match is valid as the separator does not have to be at the end of the entire text.

        Extraction rules work for multi-token strings, however, the separators between tokens are fixed by the pattern.
        For multiple separator symbols, all pattern variants must be explicitly listed.

        The resulting tagger always creates non-ambiguous layers.
        """

        self.conf_param = [
            'ruleset',
            'token_separators',
            'conflict_resolver',
            'global_decorator',
            'ignore_case']

        self.input_layers = ()
        self.output_layer = output_layer
        self.output_attributes = output_attributes or ()

        # Validate ruleset. It must exist
        if not isinstance(ruleset, Ruleset):
            raise ValueError('Argument ruleset must be of type RuleSet')
        if not (set(ruleset.output_attributes) <= set(self.output_attributes)):
            raise ValueError('Output attributes of a ruleset must match the output attributes of a tagger')

        self.ruleset = copy(ruleset)
        self.token_separators = token_separators
        self.global_decorator = global_decorator
        self.conflict_resolver = conflict_resolver
        self.ignore_case = ignore_case

        # We bypass restrictions of Tagger class to set some private attributes
        super(Tagger, self).__setattr__('_automaton', Automaton())
        super(Tagger, self).__setattr__('_attribute_map', self.ruleset.attribute_map)
        super(Tagger, self).__setattr__('_decorator_map', self.ruleset.decorator_map)
        super(Tagger, self).__setattr__('_priority_map', self.ruleset.priority_map)

        # Configures automaton to match the patters in the ruleset
        if self.ignore_case:
            for pattern in self._attribute_map:
                self._automaton.add_word(pattern.lower(), len(pattern))
            for pattern in self._decorator_map:
                self._automaton.add_word(pattern.lower(), len(pattern))
        else:
            for pattern in self._attribute_map:
                self._automaton.add_word(pattern, len(pattern))
            for pattern in self._decorator_map:
                self._automaton.add_word(pattern, len(pattern))

        self._automaton.make_automaton()

    def _make_layer(self, text: Text, layers=None, status=None):

        layer = Layer(
            name=self.output_layer,
            attributes=self.output_attributes,
            text_object=text,
            ambiguous=True
        )

        raw_text = text.text.lower() if self.ignore_case else text.text
        all_matches = self.extract_matches(raw_text, self.token_separators)

        if self.conflict_resolver == 'KEEP_ALL':
            return self.add_decorated_spans_to_layer(layer, iter(all_matches))
        elif self.conflict_resolver == 'KEEP_MAXIMAL':
            return self.add_decorated_spans_to_layer(layer, self.keep_maximal_matches(all_matches))
        elif self.conflict_resolver == 'KEEP_MINIMAL':
            return self.add_decorated_spans_to_layer(layer, self.keep_minimal_matches(all_matches))
        elif callable(self.conflict_resolver):
            return self.conflict_resolver(layer, self.iterate_over_decorated_spans(layer, iter(all_matches)))

        raise ValueError("Data field conflict_resolver is inconsistent")

    # noinspection PyUnresolvedReferences
    def extract_matches(self, text: str, separators: str) -> List[Tuple[ElementaryBaseSpan, str]]:
        """
        Returns a list of matches of the defined by the list of extraction rules that are canonically ordered:
            span[i].start <= span[i+1].start
            span[i].start == span[i+1].start ==> span[i].end < span[i + 1].end

        All matches are returned when no separator characters are specified.
        Given a list of separator symbols returns matches that do not contain of incomplete tokens.
        That is, the symbol before the match and the symbol after the match is a separator symbol.

        In both cases, matches can overlap and do not have to be maximal -- a span may be enclosed by another span.
        """

        match_tuples = []
        if len(separators) == 0:
            for loc, value in self._automaton.iter(text):
                match_tuples.append(
                    (ElementaryBaseSpan(start=loc - value + 1, end=loc + 1), text[loc - value + 1: loc + 1]))
        else:
            n = len(text)
            for loc, value in self._automaton.iter(text):
                end = loc + 1
                start = loc - value + 1

                # Check that a preceding symbol is a separator
                if start > 0 and text[start - 1] not in separators:
                    continue

                # Check that a succeeding symbol is a separator
                if end < n and text[end] not in separators:
                    continue

                match_tuples.append((ElementaryBaseSpan(start=start, end=end), text[start:end]))

        return sorted(match_tuples, key=lambda x: (x[0].start, x[0].end))

    # noinspection PyUnresolvedReferences
    def add_decorated_spans_to_layer(
            self,
            layer: Layer,
            sorted_tuples: Iterator[Tuple[ElementaryBaseSpan, str]]) -> Layer:
        """
        Adds annotations to extracted matches and assembles them into a layer.
        Annotations are added to extracted matches based on the right-hand-side of the matching extraction rule:
        * First statical rules are applied to specify fixed attributes. No spans are dropped!
        * Next the global decorator is applied to update the annotation.
        * A span is dropped when the resulting annotation is not a dictionary of attribute values.
        * Finally decorators from dynamical rules are applied to update the annotation.
        * A span is dropped when the resulting annotation is not a dictionary of attribute values.
        """

        text_object = layer.text_object
        current = next(sorted_tuples, None)
        # This hack is needed as EstNLTK wants complete attribute assignment for each annotation
        dummy_annotation = {attribute: None for attribute in self.output_attributes}

        while current is not None:
            span = Span(base_span=current[0], layer=layer)

            for attri in self._attribute_map.get(current[1], {}):
                annot = Annotation(span, **{**dummy_annotation, **attri})
                span.add_annotation(annot)

            # Drop spans for which the global decorator fails
            if self.global_decorator is not None:
                span.annotations[0] = self.global_decorator(text_object, span)
                if not isinstance(span.annotations[0], dict):
                    current = next(sorted_tuples, None)
                    continue

            # No dynamic rules to change the annotation
            decorator = self._decorator_map.get(current[1], None)
            if decorator is None:
                layer.add_span(span)
                current = next(sorted_tuples, None)
                continue

            # Drop all spans for which the decorator fails
            span.annotations[0] = decorator(text_object, span)
            if isinstance(span.annotations[0], dict):
                layer.add_span(span)

            current = next(sorted_tuples, None)

        return layer

    # noinspection PyUnresolvedReferences
    def iterate_over_decorated_spans(
            self,
            layer: Layer,
            sorted_tuples: Iterator[Tuple[ElementaryBaseSpan, str]]
    ) -> Generator[Tuple[Span, int, int], None, None]:
        """
        Returns a triple (span, group, priority) for each match that passes validation test.

        Group and priority information is lifted form the matching extraction rules.
        By construction a dynamic and static rule must have the same group and priority attributes.

        Annotations are added to extracted matches based on the right-hand-side of the matching extraction rule:
        * First statical rules are applied to specify fixed attributes. No spans are dropped!
        * Next the global decorator is applied to update the annotation.
        * A span is dropped when the resulting annotation is not a dictionary of attribute values.
        * Finally decorators from dynamical rules are applied to update the annotation.
        * A span is dropped when the resulting annotation is not a dictionary of attribute values.
        """

        text_object = layer.text_object
        current = next(sorted_tuples, None)
        # This hack is needed as EstNLTK wants complete attribute assignment for each annotation
        dummy_annotation = {attribute: None for attribute in self.output_attributes}
        while current is not None:
            span = Span(base_span=current[0], layer=layer)
            group, priority = self._priority_map.get[current[1]]
            # This hack is needed as EstNLTK wants complete attribute assignment for each annotation
            span.add_annotation(Annotation(span, **{**dummy_annotation, **self._attribute_map.get(current[1], {})}))

            # Drop spans for which the global decorator fails
            if self.global_decorator is not None:
                span.annotations[0] = self.global_decorator(text_object, span)
                if not isinstance(span.annotations[0], dict):
                    current = next(sorted_tuples, None)
                    continue

            # No dynamic rules to change the annotation
            decorator = self._decorator_map.get(current[1], None)
            if decorator is None:
                yield (span, group, priority)
                current = next(sorted_tuples, None)
                continue

            # Drop all spans for which the decorator fails
            span.annotations[0] = decorator(text_object, span)
            if isinstance(span.annotations[0], dict):
                yield (span, group, priority)

            current = next(sorted_tuples, None)

    @staticmethod
    def keep_maximal_matches(sorted_tuples: List[Tuple[ElementaryBaseSpan, str]]) \
            -> Generator[Tuple[ElementaryBaseSpan, str], None, None]:
        """
        Given a list of canonically ordered spans removes spans that are covered by another span.
        The outcome is also in canonical order.

        Recall that canonical order means:
            span[i].start <= span[i+1].start
            span[i].start == span[i+1].start ==> span[i].end < span[i + 1].end
        """

        sorted_tuples = iter(sorted_tuples)
        current_tuple = next(sorted_tuples, None)
        while current_tuple is not None:
            next_tuple = next(sorted_tuples, None)

            # Current span is last
            if next_tuple is None:
                yield current_tuple
                return

            # Check if the next span covers the current_tuple
            if current_tuple[0].start == next_tuple[0].start:
                assert current_tuple[0].end < next_tuple[0].end, "Tuple sorting does not work as expected"
                current_tuple = next_tuple
                continue

            yield current_tuple

            # Ignore following spans that are covered by the current span
            while next_tuple[0].end <= current_tuple[0].end:
                next_tuple = next(sorted_tuples, None)
                if next_tuple is None:
                    return

            current_tuple = next_tuple

    @staticmethod
    def keep_minimal_matches(sorted_tuples: List[Tuple[ElementaryBaseSpan, str]]) \
            -> Generator[Tuple[ElementaryBaseSpan, str], None, None]:
        """
        Given a list of canonically ordered spans removes spans that enclose another smaller span.
        The outcome is also in canonical order.

        Recall that canonical order means:
            span[i].start <= span[i+1].start
            span[i].start == span[i+1].start ==> span[i].end < span[i + 1].end
        """

        work_list = []
        sorted_tuples = iter(sorted_tuples)
        current_tuple = next(sorted_tuples, None)
        while current_tuple is not None:
            new_work_list = []
            add_current = True
            # Work list is always in canonical order and span ends are strictly increasing.
            # This guarantees that candidates are released in canonical order as well.
            for candidate_tuple in work_list:

                # Candidate tuple is inside the current tuple. It must be last candidate
                if current_tuple[0].start == candidate_tuple[0].start:
                    new_work_list.append(candidate_tuple)
                    add_current = False
                    break

                # No further span can be inside the candidate span
                if candidate_tuple[0].end < current_tuple[0].start:
                    yield candidate_tuple
                    continue

                # Current tuple is not inside the candidate tuple
                if candidate_tuple[0].end < current_tuple[0].end:
                    new_work_list.append(candidate_tuple)

                assert candidate_tuple[0].start < current_tuple[0].start, "Tuple sorting does not work as expected"

            # The current tuple is always a candidate for minimal match
            work_list = new_work_list + [current_tuple] if add_current else new_work_list
            current_tuple = next(sorted_tuples, None)

        # Output work list as there were no invalidating spans left
        for candidate_tuple in work_list:
            yield candidate_tuple
