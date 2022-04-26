import pandas as pd



def load_dataframe(table_name):
    dataframe = pd.read_csv(f'dataset/ml-25m/ml-25m/{table_name}.csv')
    return dataframe

def load_dataframes(table_names):
    dataframes = {table_name: load_dataframe(table_name) for table_name in table_names}
    return dataframes

if __name__ == '__main__':
    load_dataframes(["links"])
