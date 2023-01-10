import re
import os
from estnltk.taggers import TaggerTester
from cda_data_cleaning.fact_extraction.common.taggers.cancer_stage_tagger.cancer_stage_tagger import CancerStageTagger


def run_tester(type_nr: int):
    """
    Runs tests in the folder corresponding to 'type_nr'.
    Tests are based  on all 'input_type*.json' and 'target_type*.json' files.
    """

    folder_name = "type_" + str(type_nr) + "/"

    path = "cda_data_cleaning/fact_extraction/common/taggers/cancer_stage_tagger/tests/"

    # for file in os.listdir('./' + folder_name):
    for file in os.listdir(path + folder_name):
        # find input_*.json and target_*.json files
        if re.match("input_", file):
            target_file = file.replace("input", "target")

            tester = TaggerTester(
                tagger=cancer_stage_tagger, input_file=path + folder_name + file, target_file=path + folder_name + target_file
            )
            tester.load()
            tester.run_tests()


def test_type_1():
    run_tester(1)


cancer_stage_tagger = CancerStageTagger()
