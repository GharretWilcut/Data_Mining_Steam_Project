# get data from the csv to a df for us to run machine learning models
import os
import sqlite3

import pandas as pd

DATA_DIR = f"{os.path.dirname(os.path.realpath(__file__))}/../data"


def read_data(filename: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(f"{DATA_DIR}/{filename}")
    df = pd.read_sql(sql="SELECT * FROM games", con=conn)
    conn.close()
    return df


def write_data(df: pd.DataFrame, filename: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(f"{DATA_DIR}/{filename}")
    df.to_sql("games", conn, if_exists="replace", index=False)
    conn.close()
