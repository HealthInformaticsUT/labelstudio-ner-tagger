import random


def create_dataset_for_NSP(dataset, tokenizer, text_col="text", nsp_probability=0.5, **kwargs):
    """
    Creates a dataset ready to be used in the Next Sentence Prediction task.
    :param dataset: a datasets object containing 1 column with 1 document on each row. Each document consists of
     consecutive sentences that are separated by "\n".
    :param tokenizer: A tokenizer that tokenizes tokens to input_ids
    :param text_col: (str, default="text"), the column name in the dataset
    :param nsp_probability: (float, default=0.5), the probability that the next sentence is consecutive.
    :param **kwargs: (None or dict), tokenization arguments (see:
     https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
     )
    :return: a dataset with the columns: "input_ids", "attention_mask", "token_type_ids" and "next_sentence_label"
    """
    # for some reason tokenizer.__call__() method does not like kwargs. Raises an exception when an unknown keyword
    # argument is given, thus it is required to sort those out.
    possible_args = {'text_pair', 'add_special_tokens', 'padding', 'truncation', 'max_length', 'stride',
                     'is_split_into_words', 'pad_to_multiple_of', 'return_tensors', 'return_token_type_ids',
                     'return_attention_mask', 'return_overflowing_tokens', 'return_special_tokens_mask',
                     'return_offsets_mapping', 'return_length', 'verbose'}
    tokenization_args = {k: kwargs[k] for k in kwargs if k in possible_args}

    def get_next_sentence(sentence, next_sentence, paragraphs):
        if random.random() < nsp_probability:
            is_next = 1
        else:
            random_paragraph = random.choice(paragraphs)
            while random_paragraph == paragraphs:
                random_paragraph = random.choice(paragraphs)
            next_sentence = random.choice(random_paragraph)
            is_next = 0
        return sentence, next_sentence, is_next

    def tokenize_and_encode(batch):
        paragraphs = [paragraph.split("\n") for paragraph in batch[text_col]]
        sentence_pairs = []
        nsp_labels = []
        for paragraph in paragraphs:
            for i in range(len(paragraph) - 1):
                nsp_training_object = get_next_sentence(paragraph[i], paragraph[i - 1], paragraphs)
                sentence_pairs.append([nsp_training_object[0], nsp_training_object[1]])
                nsp_labels.append(nsp_training_object[2])
        tokenized_sentences = tokenizer(text=sentence_pairs, **tokenization_args)
        tokenized_sentences['next_sentence_label'] = nsp_labels
        return tokenized_sentences

    return dataset.map(tokenize_and_encode, batched=True, remove_columns=[text_col])
