import re

# Convert months from word to number
MONTHS = {
    "jaanuar": 1,
    "veebruar": 2,
    "märts": 3,
    "aprill": 4,
    "mai": 5,
    "juuni": 6,
    "juuli": 7,
    "august": 8,
    "seprember": 9,
    "oktoober": 10,
    "november": 11,
    "detsember": 12,
}

# Regex fr atomic parts of datetime strings
MACROS = {
    "hour": "([0-1][0-9]|2[0-3])",
    "minute": "[0-5][0-9]",
    "second": "[0-5][0-9]",
    "DAY": "[0-3]?[0-9]",
    "MONTH": "(0?[1-9]|1[0-2])",
    "MONTHSTR": "(%s)" % "|".join(MONTHS.keys()),
    "YEAR": "((19[0-9]{2})|(20[0-9]{2})|([0-9]{2}))",
    "LONGYEAR": "((19[0-9]{2})|(20[0-9]{2}))",
}

# Convert atomic pre-regex strings to keyed regexes
for k, v in MACROS.items():
    MACROS[k] = r"(?P<{key}>{regex})".format(key=k, regex=v)

# Specify larger regular chunks of time and store in MACROS dictionary
MACROS["DATE"] = "{DAY}((\.\s?)|(\s)){MONTH}((\.\s*)|(\s+)){YEAR}".format(**MACROS)
MACROS["DATESTR"] = "{DAY}((\.\s*)|(\s+)){MONTHSTR}((\.\s*)|(\s+)){YEAR}".format(**MACROS)
MACROS["TIME"] = "{hour}[:]{minute}(:{second})?".format(**MACROS)


# Specify prefixes encountered in the text fields of EGV Epicrises
PREFIXES = {
    "rec": r"(?:(^\s*)|(\n\s*))((?:Kokkuvõte ja soovitused)|(?:Ravi)|(?:Operatsioon)|(?:Anamnees))(?:\?|:?)\s",
    "reg": r"(?:(^\s*)|(\n\s*)|((?<=[\.\!\?])[ ]{3,}))",
}
PREFIXES["star"] = PREFIXES["reg"] + "\*"

# Format the regexes with names for later identification
REGEX = {
    "timestr": "k(el)?l\s{TIME}".format(**MACROS),
    "date_time": "{DATE}\s*{TIME}".format(**MACROS),
    "datestr_time": "{DATESTR}\s*{TIME}".format(**MACROS),
    "date_timestr": "{DATE}[.a ]+\s*k(el)?l\.*\s*{TIME}".format(**MACROS),
    "datestr_timestr": "{DATESTR}[.a ]+\s*k(el)?l\.*\s*{TIME}".format(**MACROS),
    "date": "{DATE}($|[^0-9])".format(**MACROS),
    "datestr": "{DATESTR}($|[^0-9])".format(**MACROS),
    "pdate_noyear": "{DAY}(?:\.\s?|\s){MONTH}($|\s)".format(**MACROS),
    "pdate_noday": "{MONTH}\.\s?{LONGYEAR}($|\s)".format(**MACROS),
    "pdatetime_noyear": "{DAY}\.\s?{MONTH}\s*k(el)?l\s{TIME}".format(**MACROS),
    # Matching just a year causes more problems than it solves.
    # Gets mentions of old illnesses and would cause splitting inside visit event
    #'yearstr': '{LONGYEAR}\s*a'.format(**MACROS),
    #'yearstr_dot': '{LONGYEAR}\s*\.a'.format(**MACROS),
    #'year': '{LONGYEAR}($|\s)'.format(**MACROS),
    "spec_date": "(:?[A-Z]+)\s(\(\s?{DATE}\s?\))".format(**MACROS),
    "kp_date": "Kuupäev: {DATE}".format(**MACROS),
    "kp_datetime": "Kuupäev: {DATE}\s*{TIME}".format(**MACROS),
}

regexes = {}
for reg_name, reg in REGEX.items():
    for pre_name, pre in PREFIXES.items():
        regexes[pre_name + "." + reg_name] = re.compile(pre + reg)
