# get data from the csv to a df for us to run machine learning models
import sqlite3

import pandas as pd


def get_data():
    conn = sqlite3.connect("data/steam_games_dataset.db")
    df = pd.read_sql("SELECT * FROM games", conn)
    conn.close()
    return df
