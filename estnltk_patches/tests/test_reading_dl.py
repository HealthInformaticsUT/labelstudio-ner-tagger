from pipelines.step03_BERT_fine_tuning.dataloaders import Tokens


dl = Tokens.Tsv("dl/stuckinthemiddlewithyou.tsv")
dl2 = Tokens.Tsv("dl/dage_dev_cleaned.tsv")

print(dl.read())
print(dl2.read())