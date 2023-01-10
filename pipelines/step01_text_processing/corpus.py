import csv

from estnltk.corpus_processing.parse_koondkorpus import parse_tei_corpora
from pipelines.step01_text_processing.textprocessing.parallel_workers import clean_and_extract_parallel
from .textprocessing.text_cleaning import reformat_sentences


def clean_and_extract(corp_path, out_file_path, max_processes=1, clean=None, verbose=False):
    """
    Cleans (optinal) and extract se:param corp_path: path to the corpus directory that contains .xml files
    :param corp_path: Path to the corpus
    :param out_file_path: output .txt file
    :param max_processes: The number of processes you want to create and use. Note that you don't need as many if
         you use a cheap clean function (if at all) than an expensive one.
    :param clean: Function, that cleans takes an EstNLTK object as an argument and cleans it.
    There are two pre-made cleaning functions in this package {clean_med, clean_med_r_events}
    :param verbose: to show progress or not
    """

    if max_processes >= 3:
        _clean_and_extract_par(corp_path, out_file_path, max_processes, clean, verbose)
    # if too few processes are used, then use 1 process
    else:
        with open(out_file_path, "w", newline='', encoding="utf-8") as out_file:
            tsv_writer = csv.writer(out_file, delimiter="\t")
            tsv_writer.writerow(["text"])
            for text_obj in parse_tei_corpora(corp_path, target=['artikkel']):
                if clean is not None:
                    text_obj = clean(text_obj)
                sentences = reformat_sentences(text_obj)
                tsv_writer.writerow([sentences])


def _clean_and_extract_par(corp_path, out_file_path, max_processes=3, clean=None, verbose=False):
    """
        Cleans (optinal) and extract sentences from a corpus.
        :param corp_path: path to the corpus directory that contains .xml files
        :param out_file_path: path to the output .txt file
        :param max_processes: The number of processes you want to create and use. Note that you don't need as many if
         you use a cheap clean function (if at all) than an expensive one.
        :param clean: Function, that cleans takes an EstNLTK object as an argument and cleans it.
        There are two pre-made cleaning functions in this package {clean_med, clean_med_r_events}
        """
    clean_and_extract_parallel(_read_corpus, (corp_path,), out_file_path, max_processes, clean, verbose=verbose)


def _read_corpus(recorded_data, corpus_path):
    """
    A function that reads texts from a tsv file and puts them into multiprocessing Queue "recorder_data"
    :param recorded_data: A queue shared by this and text processing threads
    :param corpus_path:
    """
    for text_obj in parse_tei_corpora(corpus_path, target=['artikkel']):
        recorded_data.put(text_obj)
