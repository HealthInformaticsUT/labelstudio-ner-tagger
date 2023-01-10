import torch
from transformers import BertTokenizer, TextDatasetForNextSentencePrediction, DataCollatorForLanguageModeling


class PreTrainingDataset(torch.utils.data.Dataset):

    def __init__(self, vocab_path, input_files, lowercase=False, mlm=True, mlm_prob=0.15, nsp=True):
        tokenizer = BertTokenizer(vocab_path)
        self.encodings = []
        self.labels = []


    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
