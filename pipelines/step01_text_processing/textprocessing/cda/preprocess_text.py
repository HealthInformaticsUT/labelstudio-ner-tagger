import re


def preprocess_text(text):
    """
    Preprocessing and unifying text field/freetext content before giving it to taggers.

    Some texts contain thousands of "\n", replace them with 4 "\n".
    Less is not recommended, because some taggers depend on multiple newlines
    (https://git.stacc.ee/project4/cda-data-cleaning/tree/master/cda_data_cleaning/data_cleaning/analysis_data_cleaning/step03_create_and_clean_analysis_texts/taggers)
    """
    return re.sub("(\n\s*\n\s*\n\s*\n\s*\n)+", "\n\n\n\n", text)
