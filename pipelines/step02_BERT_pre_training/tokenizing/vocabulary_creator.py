from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.pre_tokenizers import Whitespace


def create_vocabulary(model_path, files, size=8000, special_tokens=None, additional_special_tokens=None, lowercase=False, **kwargs):
    """
    Creates a vocabulary by using the SentencePiece tokenizer in unigram mode.

    :param vocab_path: (The path with the name of the resulting vocabulary) type: std::string
    :param files: (list of input text files) type: [std::string],
    :param size: (vocabulary size)  type: int32 default: 8000
    :param special_tokens: (Special symbols). type: [std::string] default: ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    :param additional_special_tokens:  (Additional special symbols). type: [std::string] default: None
    :param lowercase: (To lowercase the tokens or not) type: boolean default: False
    :return: NoneType
    """
    if special_tokens is None:
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    if additional_special_tokens is not None:
        special_tokens.extend(additional_special_tokens)

    tokenizer = BertWordPieceTokenizer(lowercase=lowercase)
    tokenizer.train(files, vocab_size=size, special_tokens=special_tokens)
    tokenizer.save_model(model_path)

