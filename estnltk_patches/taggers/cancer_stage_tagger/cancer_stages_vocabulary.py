import re


def clean_stage(input_stage):
    number_dict = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV'}
    cleaned = "".join(input_stage.split(" ")).upper()
    for i in number_dict:
        cleaned = cleaned.replace(i, number_dict[i])
    return cleaned


def fix_casing(code, letter):
    if letter in code.upper():
        return code.upper().split(letter)[0].lower() + letter + code.upper().split(letter)[1].upper()
    return None


def pick_gleason(input_gleason):
    output = re.sub(r'[^\dx+=]', '', input_gleason)
    return output


def pick_code(code):
    if code is None:
        return None

    T_value, N_value, M_value, G_value = None, None, None, None

    # removing anything between parentheses, mostly to get rid of ANONYMOUS tags
    anon_regex = r"\(.*\)"
    anon_tag = re.search(anon_regex, code)
    if anon_tag is not None:
        code = code.replace(anon_tag[0], "")

    # lets remove any gleason tags, which just add noise
    gleason_regex = r"(Gleason|GS|Gl).*?((\d ?\+ ?\d ?\= ?)|(\d\+))?\d{1,2}"
    gleason_tag = re.search(gleason_regex, code)
    if gleason_tag is not None:
        code = code.replace(gleason_tag[0], "")

    for i in [".", ",", ";", "-", "_"]:
        code = code.replace(i, " ")

    # replacing character 'o'-s with number 0-s.
    for i in ["o", "O"]:
        code = code.replace(i, "0")

    T_reg = r"[p|c|r|y]{0,4} ?T(x|X|o|O|4|0|([1-3]([a-cA-C])?))"
    T_reg_value = re.search(T_reg, code)
    if T_reg_value is not None:
        code = code.replace(T_reg_value[0], "")
        T_value = fix_casing(T_reg_value[0].replace(" ", ""), "T")

    N_reg = r"[p|c|r|y]{0,4} ?N(x|X|o|O|4|0|([1-3]([a-cA-C])?))"
    N_reg_value = re.search(N_reg, code)
    if N_reg_value is not None:
        code = code.replace(N_reg_value[0], "")
        N_value = fix_casing(N_reg_value[0].replace(" ", ""), "N")

    M_reg = r"[p|c|r|y]{0,4} ?M(x|X|o|O|4|0|([1-3]([a-cA-C])?))"
    M_reg_value = re.search(M_reg, code)
    if M_reg_value is not None:
        code = code.replace(M_reg_value[0], "")
        M_value = fix_casing(M_reg_value[0].replace(" ", ""), "M")

    G_reg = r"G\d"
    G_reg_value = re.search(G_reg, code)
    if G_reg_value is not None:
        G_value = G_reg_value[0].replace(" ", "")

    output_string = ", ".join([str(T_value), str(N_value), str(M_value)])
    if G_value is not None:
        output_string += ", " + str(G_value)

    return output_string


vocabulary = [
    {
        "grammar_symbol": "STAGE",
        "regex_type": "r1",
        "_regex_pattern_": r"(I[IV]* *[ABCabc]) *(st)?",
        "_group_": 0,
        "_priority_": 0,
        "_validator_": lambda m: True,
        "value": lambda m: clean_stage(m.group(1).strip()),
    },
    {
        "grammar_symbol": "STAGE",
        "regex_type": "r2",
        "_regex_pattern_": r"(I[IV]*) *st",
        "_group_": 0,
        "_priority_": 0,
        "_validator_": lambda m: True,
        "value": lambda m: clean_stage(m.group(1).strip()),
    },
    {
        "grammar_symbol": "STAGE",
        "regex_type": "r3",
        "_regex_pattern_": r"([0-5] *[ABCabc]) *st",
        "_group_": 0,
        "_priority_": 0,
        "_validator_": lambda m: True,
        "value": lambda m: clean_stage(m.group(1).strip()),
    },
    {
        "grammar_symbol": "STAGE",
        "regex_type": "r4",
        "_regex_pattern_": r"([0-5]) *st",
        "_group_": 0,
        "_priority_": 0,
        "_validator_": lambda m: True,
        "value": lambda m: clean_stage(m.group(1).strip()),
    },
    {
        "grammar_symbol": "STAGE",
        "regex_type": "r5",
        "_regex_pattern_": r"st\.? *([0-5] *[ABCabc]*)",
        "_group_": 0,
        "_priority_": 0,
        "_validator_": lambda m: True,
        "value": lambda m: clean_stage(m.group(1).strip()),
    },
    {
        "grammar_symbol": "STAGE",
        "regex_type": "r6",
        "_regex_pattern_": r"st\.? *(I[IV]* *[ABCabc]*)",
        "_group_": 0,
        "_priority_": 0,
        "_validator_": lambda m: True,
        "value": lambda m: clean_stage(m.group(1).strip()),
    },
    {
        "grammar_symbol": "Gleason",
        "regex_type": "gleason",
        "_regex_pattern_": r"(Gleason|GS|Gl).{0,3}?((\d ?\+ ?\d ?\= ?)|(\d\+))?\d{1,2}",
        "_priority_": 0,
        "_group_": 0,
        "_validator_": lambda m: True,
        "value": lambda m: pick_gleason(m.group(0))
    },
{
        "grammar_symbol": "TNM",
        "regex_type": "tnm",
        "_regex_pattern_": r"(G\d.*)?[p|c|r|y]{0,4} ?T(x|X|o|O|(\d([a-cA-C])?)).{0,2}[p|c|r|y]{0,4} ?N(x|X|o|O|(\d([a-cA-C])?)).{0,2}[p|c|r|y]{0,4} ?M(x|X|o|O|(\d([a-cA-C])?))(.*G\d)?",
        "_priority_": 0,
        "_group_": 0,
        "_validator_": lambda m: True,
        "value": lambda m: pick_code(m.group(0))
    }
]
