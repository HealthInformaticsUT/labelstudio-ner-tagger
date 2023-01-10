from estnltk.vabamorf.morf import synthesize


def getSpanRangesR(textObj, layer, priority=0, replace_by="", textAsList=False):
    return (
        [{"start": i.start, "end": i.end, "replaced": i.text[0] if textAsList else i.text, "replaced_by": replace_by,
          "priority": priority} for i
         in
         textObj[layer].spans])


def getSpanRangesF(textObj, layer, priority=0, replace_by_f=None):
    return (
        [{"start": i.start, "end": i.end, "replaced": i.text, "replaced_by": replace_by_f(i), "priority": priority}
         for i
         in textObj[layer].spans])


def get_anon_replacement(row):
    form = row['form'][0]
    pos = row['partofspeech'][0]
    # https://estnltk.github.io/estnltk/1.4/tutorials/morf_tables.html#postag-table
    replacement = '<XXX>'
    if (pos != '?' and pos != ''):
        if (form == 'S'):  # noun
            s = synthesize('tema', pos)
            if len(s) > 0:
                replacement = s[0]
            else:
                replacement = 'ta'

        elif (form == 'V'):  # verb
            s = synthesize('tegema', pos)
            if len(s) > 0:
                replacement = s[0]
            else:
                replacement = 'tegema'

        elif (form == 'A'):  # adjective
            replacement = '<ADJ>'
        elif (form == 'H'):  # real name
            replacement = '<NAME>'

        elif (form == 'D'):  # adverb
            replacement = '<ADV>'
        elif (form == 'I'):  # interjektsioon
            replacement = '<INJ>'

    return replacement


def get_number_replacement(row):
    if row["grammar_symbol"][0] == 'NUMBER':
        v = row['value'][0]
        if isint(v):
            return "<INT>"
        return "<FLOAT>"

    return "<DATE>"


def isint(x):
    try:
        a = float(x)
        b = int(a)
    except (TypeError, ValueError):
        return False
    else:
        return a == b
