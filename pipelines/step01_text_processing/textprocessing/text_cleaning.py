import re

from estnltk import Text

# Table taggers
from .cda.preprocess_text import preprocess_text
from .cda.table.segments_tagger_all_types import SegmentsTagger
from .cda.table.type1_start_end_tagger import Type1StartEndTagger
from .cda.table.type2_start_end_tagger import Type2StartEndTagger
from .cda.table.type3_start_end_tagger import Type3StartEndTagger
from .cda.table.type4_start_end_tagger import Type4StartEndTagger
from .cda.table.type5_start_end_tagger import Type5StartEndTagger
from .cda.table.type6_start_end_tagger import Type6StartEndTagger
from .cda.table.type7_start_end_tagger import Type7StartEndTagger

# Event taggers
from .cda.events.event_header_tagger import EventHeaderTagger
from .cda.events.event_segments_tagger import EventSegmentsTagger
from .cda.events.event_token_tagger import EventTokenTagger

# Anon tagger
from .cda.anon.anonymised_tagger import AnonymisedTagger

# Dates and Numbers
from .cda.numbers.robust_date_number_tagger import RobustDateNumberTagger

from .layer_modifications import getSpanRangesR, getSpanRangesF, \
    get_number_replacement, get_anon_replacement

from .span_merging import mergeSpans

type1_tagger = Type1StartEndTagger()
type2_tagger = Type2StartEndTagger()
type3_tagger = Type3StartEndTagger()
type4_tagger = Type4StartEndTagger()
type5_tagger = Type5StartEndTagger()
type6_tagger = Type6StartEndTagger()
type7_tagger = Type7StartEndTagger()
seg_tagger = SegmentsTagger()

# Event taggers
event_token_tagger = EventTokenTagger(output_layer='event_tokens')
event_header_tagger = EventHeaderTagger()
event_segments_tagger = EventSegmentsTagger()

# Anon tagger
anon_tagger = AnonymisedTagger()

# Date and Number Tagger
date_num_tagger = RobustDateNumberTagger()


def replace_row_change_symbols(t):
    return Text(t.text.replace("\n", " <br> "))


def get_table_spans(t, priority):
    type1_tagger.tag(t)
    type2_tagger.tag(t)
    type3_tagger.tag(t)
    type4_tagger.tag(t)
    type5_tagger.tag(t)
    type6_tagger.tag(t)
    type7_tagger.tag(t)
    seg_tagger.tag(t)
    return getSpanRangesR(t, "printout_segments", priority, "")


def get_event_header_spans(t, priority):
    t = event_token_tagger(t)
    t = event_header_tagger(t)
    t = event_segments_tagger(t)
    return getSpanRangesR(t, "event_headers", priority, "", textAsList=True)


def get_anonymised_spans(t, priority):
    t = anon_tagger(t)
    return getSpanRangesF(t, "anonymised", priority, get_anon_replacement)


def get_number_spans(t, priority):
    t = date_num_tagger(t)
    return getSpanRangesF(t, "dates_numbers", priority, get_number_replacement)


def preprocess_to_estnltk_Text(text):
    text = text.strip()
    text = preprocess_text(text)
    return Text(text)


def extract_segments(t):
    event_token_tagger.tag(t)
    event_header_tagger.tag(t)
    event_segments_tagger.tag(t)

    # removing very small
    return [i for i in t.event_segments.text if len(i) > 2]


def extract_span_ranges(t):
    # spans = get_table_spans(t, 0)
    spans = get_anonymised_spans(t, 1)
    # spans = mergeSpans(spans, get_anonymised_spans(t, 1))
    spans = mergeSpans(spans, get_number_spans(t, 2))
    return spans


def clean_text_with_spans(text, spans):
    new_text = ""
    start = 0
    for span in spans:
        if span["start"] == span["end"]:
            continue
        new_text += text[start:span["start"]]
        new_text += span["replaced_by"]
        start = span["end"]
    new_text += text[start:len(text)]
    return new_text

def remove_excess_br(text):
    # removing duplicates
    text = re.sub("(<br> *)+", " <br> ", text)
    # removing from start
    text = re.sub("^ *<br> +", "", text)
    # removing from end
    text = re.sub(" *<br> *$", "", text)
    return text


"""
def clean(t):
    spans = extract_span_ranges(t)
    return clean_text_with_spans(t.text, spans)
"""


def remove_beginning_symbols(text):
    while len(text) > 0 and text[0] in {'*', '.', '-', ':', '|', ' '}:
        text = text[1:]
    return text


def reformat_sentences(t):
    if not isinstance(t, list):
        t = [t]

    sentences = []
    for text_obj in t:
        text_obj.tag_layer(['tokens', 'words', 'sentences'])
        s = []
        for sentence in text_obj.sentences:
            i = " ".join(sentence.text)
            # if sentence does not end with ".", "!" or "?", then add another one
            # Also adding very small sentences
            i = remove_beginning_symbols(i)
            # removing excess line symbols
            i = remove_excess_br(i)
            # removing duplicated spaces
            i = re.sub(' +', ' ', i)
            if len(s) != 0 and (len(s[-1]) != 0 and s[-1][-1] not in {".", "!", "?"} or len(sentence.text) < 3):
                s[-1] += i
            else:
                s.append(i)

        sentences.append("\n".join(s))
    return "\n".join(sentences)


def clean_med(t):
    t = replace_row_change_symbols(t)
    t.tag_layer(['tokens', 'words'])
    spans = extract_span_ranges(t)
    cleaned_text = clean_text_with_spans(t.text, spans)
    t_cleaned = preprocess_to_estnltk_Text(cleaned_text)
    return t_cleaned


def clean_med_events(t):
    cleaned = []
    for segment in extract_segments(t):
        cleaned.append(clean_med(Text(segment)))
    return cleaned
