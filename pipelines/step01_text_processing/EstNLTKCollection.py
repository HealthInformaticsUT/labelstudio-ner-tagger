import csv
import multiprocessing
import os
from typing import Text

from estnltk.storage import PostgresStorage
from estnltk.storage.postgres import BlockQuery
from joblib._multiprocessing_helpers import mp

from pipelines.step01_text_processing.textprocessing.text_cleaning import reformat_sentences, clean_med_events, \
    clean_med


def clean_and_extract(connection_info, table_name, out_file_path, max_workers=1, clean=None, verbose=False):
    """
    Cleans and extracts texts from a estNLTK Collection and puts these into a format used to pre-train a BERT model
    :param connection_info: (dict) parameters used to make a connection to the database
    for example:
                        {
                        "host": "localhost",
                       "port": 5432,
                       "dbname": 'postgres',
                       "user": "postgres",
                       "password": "admin",
                       "schema": 'step01_collection_test',
                       "temporary": False
                       }
    :param table_name: (str) Table/relation/collection name in the database, that contains texts
    :param out_file_path: (str) path to the output .tsv file
    :param max_workers: The number of processes you want to create and use. Note that you don't need as many if
     you use a cheap clean function (if at all) than an expensive one.
    :param clean: Function, that cleans takes an EstNLTK object as an argument and cleans it.
    There are two pre-made cleaning functions in this package {clean_med, clean_med_r_events}
    :param verbose: to show progress or not
    :return:
    """
    if clean == "clean_med":
        clean = clean_med

    elif clean == "clean_med_r_events":
        clean = clean_med_events

    max_workers = int(max_workers)

    max_workers = max_workers if mp.cpu_count() > max_workers else mp.cpu_count()
    while max_workers < 3:
        max_workers += 1
    max_workers -= 2

    return_data = mp.Queue(10000)
    jobs = []
    for r in range(max_workers):
        job = mp.Process(target=_process, args=(connection_info, table_name, max_workers, r, clean, return_data))
        jobs.append(job)
        job.start()

    end = Text("<END>")
    with open(out_file_path, "w", newline='', encoding="utf-8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter="\t")
        tsv_writer.writerow(["text"])
        while True:
            res = return_data.get()
            if res == end:
                max_workers -= 1
                if max_workers == 0:
                    break
            tsv_writer.writerow([res])

    for i in jobs:
        i.join()
        if verbose:
            os.system("echo a processing thread finished")


def _process(connection_info, table_name, m, r, clean, return_data):
    storage = PostgresStorage(**connection_info)
    collection = storage[table_name]
    for key, txt in collection.select(BlockQuery(m, r)):
        if clean is not None:
            txt = clean(txt)
        return_data.put(reformat_sentences(txt))

    return_data.put(Text("<END>"))


if __name__ == '__main__':
    conn_info = {"host": "localhost",
                       "port": 5432,
                       "dbname": 'postgres',
                       "user": "postgres",
                       "password": "admin",
                       "schema": 'step01_collection_test',
                       "temporary": False}

    clean_and_extract(conn_info, 'horisont_collection', "../../data/collection_clean_test.tsv", max_workers=2,
                      clean=None, verbose=True)
