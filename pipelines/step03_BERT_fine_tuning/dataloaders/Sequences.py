from dataclasses import dataclass
from typing import Union, List

from datasets import load_dataset, Dataset
from estnltk.storage import PostgresStorage


@dataclass(init=True)
class Tsv:
    """Reads sequences from a tsv file into a datasets class

    Args:
        :param paths (Union[str, List[str]]): a path or a list of paths to the .tsv file(s) containg sequences in column X and labels in column y
        :param X (str = "text"): The name of the column, that contains texts
        :param y (str = "y"): The name of the response variable column
        :param delimiter (str = "\t"): The delimiter used in the .tsv file
        :param skiprows (int = 0): The number of rows you'd like to skip

    :returns A dataset with columns X and y
    """

    paths: Union[str, List[str]]
    X: str = "text"
    y: str = "y"
    delimiter: str = "\t"
    skiprows: int = 0

    def read(self):
        ds = load_dataset("csv", data_files={"train": self.paths},
                          skiprows=self.skiprows, delimiter=self.delimiter)['train']
        if self.X != "X":
            ds = ds.rename_column(self.X, "X")
        if self.y != "y":
            ds = ds.rename_column(self.y, "y")
        return ds


@dataclass(init=True)
class EstNLTKCol:
    """Reads sequences from a EstNLTK collection (postgres table/relation) into a datasets class

        Args:
            :param db_con (dict):  # a dictionary containing info
             that is used to connect to the database, for example:
             {
               "host": "localhost",
               "port": 5432,
               "dbname": 'postgres',
               "user": "postgres",
               "password": "admin",
               "schema": 'step01_collection_test',
               "temporary": False
                },
            :param table_name (str): The name of the table/relation/collection to be used
            :param y (str = "y"): The name of the response variable in metadata

    :returns A dataset with columns X and y
    """
    db_con: dict
    table_name: str
    y: str = "y"

    def read(self):
        ds = {"X": [], "y": []}
        storage = PostgresStorage(**self.db_con)
        collection = storage[self.table_name]
        for k, txt, lab in collection.select(collection_meta=[self.y]):
            ds["X"].append(txt.text)
            ds["y"].append(lab[self.y])
        return Dataset.from_dict(ds)
