import pandas as pd
from cda_data_cleaning.common.db_operations import insert_to_database


class TableBuffer:
    def __init__(self, conn, schema, table_name, buffer_size, column_names, input_type):
        self.buffer_size = buffer_size
        self.conn = conn
        self.schema = schema
        self.table_name = table_name
        self.column_names = column_names
        self.result_dict_list = [] if input_type == "dict" else pd.DataFrame()
        # this will either
        # 1) contain dicts and in the end be converted to pandas and sent to db
        # 2) pandas dataframe

    def append(self, data):
        """
        List of dictinories that will be added to the final output in a dictonary format.

        e.g. data can be a list of dictionaries
        a = [{'col1':'0', 'col2':'1'}]
        b = [{'col1':'2','col2':'3'}]
        [{'col1': '0', 'col2': '1'}, {'col1': '2', 'col2': '3'}]
        """
        if type(data) == list:
            self.result_dict_list.extend(data)  # append list of dictonarys to results
        elif type(data) == pd.DataFrame:
            self.result_dict_list = pd.concat((self.result_dict_list, data), axis=0)

        if len(self.result_dict_list) >= self.buffer_size:  # list has gotten too big => send to database
            self.flush()

    def extend(self):
        print()

    def flush(self):
        """
        Converting results to pandas dataframe and sending them to database.
        """
        if len(self.result_dict_list):
            if type(self.result_dict_list) == list:
                df_final = pd.DataFrame.from_dict(self.result_dict_list)[self.column_names]
                insert_to_database(df_final, self.schema + "." + self.table_name, self.conn)
                self.result_dict_list = []  # empty the list
            elif type(self.result_dict_list) == pd.DataFrame:
                insert_to_database(self.result_dict_list, self.schema + "." + self.table_name, self.conn)
                self.result_dict_list = pd.DataFrame()  # empty the list

    def close(self):
        self.conn.close()
