import csv
import os

from estnltk import Text
from joblib._multiprocessing_helpers import mp

from .text_cleaning import reformat_sentences, clean_med_events, clean_med


def clean_and_extract_parallel(data_input_f, data_input_args, out_path, max_processes=3, clean=None, verbose=False):
    """
    A parallelized pipeline that converts and cleans text from a source tsv file into usable form for training a BERT model.
    :param data_input_f:
    :param data_input_args:
    :param out_path: path to the output .txt file
    :param max_processes: The number of processes you want to create and use. Note that you don't need as many if
     you use a cheap clean function (if at all) than an expensive one.
    :param clean: Function, that cleans takes an EstNLTK object as an argument and cleans it.
    There are two pre-made cleaning functions in this package {clean_med, clean_med_r_events}
    :param verbose: to show progress or not
    """
    if clean == "clean_med":
        clean = clean_med

    elif clean == "clean_med_r_events":
        clean = clean_med_events

    max_processes = int(max_processes)

    threads = max_processes if mp.cpu_count() > max_processes else mp.cpu_count()
    while threads < 3:
        threads += 1
    threads -= 2
    if verbose:
        os.system("echo max process_count: " + str(mp.cpu_count()))
        os.system("echo using 1 for inputstream, 1 for outputstream and " + str(threads) + " for processing")

    recorded_data = mp.Queue(10000)
    return_data = mp.Queue(10000)

    producer = mp.Process(target=data_input_f, args=(recorded_data, *data_input_args))
    producer.start()

    jobs = []
    for i in range(threads):
        job = mp.Process(target=_process, args=(recorded_data, clean, return_data))
        jobs.append(job)
        job.start()

    writer = mp.Process(target=_write, args=(return_data, out_path))
    writer.start()

    producer.join()
    for i in range(threads):
        recorded_data.put(Text("<END>"))
    recorded_data.close()

    if verbose:
        os.system("echo file-reader finished")

    for i in jobs:
        i.join()
        if verbose:
            os.system("echo a processing thread finished")

    return_data.put("<END>")
    return_data.close()
    writer.join()
    if verbose:
        os.system("echo Writer finished")


def _process(recorded_data, clean, return_data):
    end = Text("<END>")
    while True:
        text_obj = recorded_data.get()
        if text_obj == end:
            break
        if clean is not None:
            text_obj = clean(text_obj)
        return_data.put(reformat_sentences(text_obj))


def _write(return_data, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8') as out:
        tsv_writer = csv.writer(out, delimiter="\t")
        tsv_writer.writerow(["text"])
        while True:
            res = return_data.get()
            if res == "<END>":
                break
            tsv_writer.writerow([res])
