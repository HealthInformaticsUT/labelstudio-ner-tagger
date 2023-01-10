import csv

from estnltk import Text

from .textprocessing.parallel_workers import clean_and_extract_parallel
from .textprocessing.text_cleaning import reformat_sentences, clean_med_events, clean_med


def clean_and_extract(in_tsv_path, out_path, text_col_i=1, max_processes=1, clean=None, tsv_newline='',
                      tsv_encoding='utf-8', tsv_delimiter="\t", verbose=False):
    """
        A pipeline that converts and cleans text from a source tsv file into usable form for training a BERT model.
        :param in_tsv_path: path to the source .tsv file
        :param out_path: path to the output .txt file
        :param text_col_i: index of the column (zero-based), that contains text to be cleaned and extracted
        :param max_processes: The number of processes you want to create and use. Note that you don't need as many if
         you use a cheap clean function (if at all) than an expensive one.
        :param clean: Function, that cleans takes an EstNLTK object as an argument and cleans it.
        There are two pre-made cleaning functions in this package {clean_med, clean_med_r_events}
        :param tsv_newline: default = '', used for output file (input newline assumed to be '')
        :param tsv_encoding: default = 'utf-8', used for output file (input file assumed to be in utf-8)
        :param tsv_delimiter: default = "\t", used for input and output file, both
        :param verbose: to show progress or not
        """
    if max_processes >= 3:
        _clean_and_extract_par(in_tsv_path, out_path, text_col_i,
                               max_processes=max_processes, clean=clean, tsv_newline=tsv_newline,
                               tsv_encoding=tsv_encoding, tsv_delimiter=tsv_delimiter, verbose=verbose)
    else:
        _clean_and_extract(in_tsv_path, out_path, text_col_i, clean=clean, tsv_newline=tsv_newline,
                           tsv_encoding=tsv_encoding, tsv_delimiter=tsv_delimiter)


# single process
def _clean_and_extract(in_tsv_path, out_path, text_col_i=1, clean=None, tsv_newline='',
                       tsv_encoding='utf-8', tsv_delimiter="\t"):
    text_col_i = int(text_col_i)

    if clean == "clean_med":
        clean = clean_med

    elif clean == "clean_med_r_events":
        clean = clean_med_events

    file = open(in_tsv_path, newline='', encoding='utf-8')
    out_file = open(out_path, 'w', newline=tsv_newline, encoding=tsv_encoding)
    tsv_reader = csv.reader(file, delimiter=tsv_delimiter)
    tsv_writer = csv.writer(out_file, delimiter=tsv_delimiter)

    for i, row in enumerate(tsv_reader):
        if i == 0:
            tsv_writer.writerow([row[text_col_i]])
            continue

        if len(row) < text_col_i:
            continue

        text_obj = Text(row[text_col_i])
        if clean is not None:
            text_obj = clean(text_obj)
        sentences = reformat_sentences(text_obj)
        tsv_writer.writerow([sentences])

    file.close()
    out_file.close()


# multiprocess
def _clean_and_extract_par(tsv_path, out_path, text_col_i=1,
                           max_processes=3, clean=None, tsv_newline='',
                           tsv_encoding='utf-8', tsv_delimiter="\t", verbose=False):
    clean_and_extract_parallel(_read_tsv, (tsv_path, text_col_i, tsv_newline, tsv_encoding, tsv_delimiter),
                               out_path, max_processes, clean, verbose)


def _read_tsv(recorded_data, tsv_path, text_col_i, newline='', encoding='utf-8', delimiter="\t"):
    """
    A function that reads texts from a tsv file and puts them into multiprocessing Queue "recorder_data"
    :param recorded_data: A queue shared by this and text processing threads
    :param tsv_path: path to the source .tsv file
    :param text_col_i: index of the column, that contains text to be cleaned and extracted
    :param newline: default = \t
    :param encoding: default = utf-8
    :param delimiter: default = ''
    """
    with open(tsv_path, newline=newline, encoding=encoding) as file:
        tsv_reader = csv.reader(file, delimiter=delimiter)
        next(tsv_reader)
        for row in tsv_reader:
            if len(row) < text_col_i:
                continue
            recorded_data.put(Text(row[text_col_i]))
