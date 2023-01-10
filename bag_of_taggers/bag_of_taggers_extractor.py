from estnltk_patches.taggers.robust_date_number_tagger.robust_date_number_tagger import RobustDateNumberTagger
from estnltk_patches.span_extractor_base.RetrainableSpanExtractor import RetrainableSpanExtractor
from estnltk_patches.taggers.cancer_stage_tagger.cancer_stage_tagger import CancerStageTagger
from estnltk_patches.taggers.ner_tagger.ner_tagger import BertNerTagger

from estnltk import Tagger

from typing import List

import pathlib


class BOTwithoutSub(RetrainableSpanExtractor):

    @property
    def pretrained_model_path(self):
        return pathlib.Path(__file__).parent.resolve() / '../medbert_models/pretrained_model'

    def _initalize_retrainable_tagger(self, pretrained_model_path: str) -> BertNerTagger:
        # raise NotImplementedError('Define a retrainable BERT tagger')
        return BertNerTagger(
            pretrained_model_path=self.pretrained_model_path,
            annotations=['B-SYMP', 'B-DRUG']
        )

    def _initialize_finetuned_tagger(self, pretrained_model_path: str, finetuned_model_path: str) -> BertNerTagger:
        # raise NotImplementedError('Define a fine-tuned tagger used for predicting annotations')
        finetuned_model_path = self.train_output['finetuned_model_path']
        return BertNerTagger(
            pretrained_model_path=self.pretrained_model_path,
            finetuned_model_path=finetuned_model_path,
            annotations=['B-SYMP', 'B-DRUG']
        )

    def _initialize_markup_taggers(self) -> List[Tagger]:
        tagger1 = RobustDateNumberTagger()
        tagger2 = CancerStageTagger()
        return [tagger1, tagger2]
