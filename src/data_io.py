# get data from the csv to a df for us to run machine learning models
import sqlite3

import pandas as pd

FILE = "data/steam_games_dataset.db"


def get_data():
    conn = sqlite3.connect(FILE)
    df = pd.read_sql("SELECT * FROM games", conn)
    conn.close()
    return df


def write_data(df: pd.DataFrame):
    conn = sqlite3.connect(FILE)
    df.to_sql("games", conn, if_exists="replace")
    conn.close()
