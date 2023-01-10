import csv
from dataclasses import dataclass
from typing import Union, List

from datasets import Dataset
from estnltk.layer import AmbiguousAttributeList
from estnltk.storage import PostgresStorage


@dataclass(init=True)
class Tsv:
    """Reads sequences from a tsv file into a datasets class

    Args:
        :param paths (Union[str, List[str]]): An integer.
        :param has_header_col (bool = True): Set to True, if there is a header row in the dataset
        :param X (Union[str, int] = "text"): The name or index of the column, that contains texts
        :param y (Union[str, int] = "y"): The name or index of the response variable column
        :param delimiter (str = "\t"): The delimiter used in the .tsv file
        :param skiprows (int = 0): The number of rows you'd like to skip

    :returns A dataset with columns X and y
    """

    paths: Union[str, List[str]]
    has_header_col: bool = True
    X: Union[str, int] = "text"
    y: Union[str, int] = "y"
    delimiter: str = "\t"
    skiprows: int = 0

    def read(self):
        if not isinstance(self.paths, list):
            self.paths = [self.paths]
        ds = {"X": [], "y": []}
        if self.paths[0] == "":
            return ds

        for path in self.paths:
            x, y = self._read_BIO_format(path)
            ds['X'].extend(x)
            ds['y'].extend(y)
        return Dataset.from_dict(ds)

    def _read_BIO_format(self, file_path):
        sentences = []
        labels = []
        tokens = []
        labs = []
        with open(file_path, encoding="utf-8") as input_file:
            reader = csv.reader(input_file, delimiter=self.delimiter)

            if self.has_header_col and isinstance(self.X, str) and isinstance(self.y, str):
                tok_i, lab_i = self._resolve_ds_col_ids(next(reader))
            elif isinstance(self.X, int) and isinstance(self.y, int):
                tok_i, lab_i = self.X, self.y
            else:
                raise ValueError("Invalid combination of has_header_col, text_col and y_col")

            for _ in range(self.skiprows):
                next(reader)
            for row in reader:
                if row[0] == '':
                    sentences.append(tokens)
                    labels.append(labs)
                    tokens = []
                    labs = []
                else:
                    tokens.append(row[tok_i])
                    labs.append(row[lab_i])

        return sentences, labels

    def _resolve_ds_col_ids(self, row):
        return self._resolve_ds_col_id(row, self.X), self._resolve_ds_col_id(row, self.y)

    def _resolve_ds_col_id(self, row, col):
        if isinstance(col, int):
            return col
        for i, n in enumerate(row):
            if n.strip() == col:
                return i
        return None


def _flatten(x):
    if isinstance(x, AmbiguousAttributeList):
        return [i[0] for i in x]
    return x


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
            :param layer (str): name of the layer that contains both X and y
            :param X (str = "text"): The name of the column, that contains texts
            :param y (str = "y"): The name of the response variable column

    :returns A dataset with columns X and y
    """
    db_con: dict
    table_name: str
    layer: str
    X: str = "text"
    y: str = "y"

    def read(self):
        ds = {"X": [], "y": []}
        storage = PostgresStorage(**self.db_con)
        collection = storage[self.table_name]
        for t in collection:
            ds["X"].append(_flatten(t[self.layer][self.X]))
            ds["y"].append(_flatten(t[self.layer][self.y]))
        return Dataset.from_dict(ds)

