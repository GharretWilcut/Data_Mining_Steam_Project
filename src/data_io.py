# get data from the csv to a df for us to run machine learning models
import sqlite3

import pandas as pd


def read_data(location: str):
    conn = sqlite3.connect(location)
    df = pd.read_sql(sql="SELECT * FROM games", con=conn)
    conn.close()
    return df


def write_data(df: pd.DataFrame, location: str):
    conn = sqlite3.connect(location)
    df.to_sql("games", conn, if_exists="replace", index=False)
    conn.close()
