import sqlite3

import pandas as pd

df = pd.read_parquet(
    "hf://datasets/FronkonGames/steam-games-dataset/data/train-00000-of-00001.parquet"
)

# Export to SQLite database
conn = sqlite3.connect("data/steam_games_dataset.db")
df.to_sql("games", conn, if_exists="replace")
conn.close()
