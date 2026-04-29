import json

import pandas as pd

from data_io import write_data

if __name__ == "__main__":
    df = pd.read_parquet(
        "hf://datasets/FronkonGames/steam-games-dataset/data/train-00000-of-00001.parquet"
    )

    # Convert list columns to stringified lists to store in SQL database correctly
    for list_column in (
        "supported_languages",
        "full_audio_languages",
        "developers",
        "publishers",
        "categories",
        "genres",
        "tags",
        "screenshots",
        "movies",
    ):
        df[list_column] = df[list_column].map(lambda x: ",".join(x))
    df["packages"] = df["packages"].map(lambda x: json.dumps(x))

    write_data(df, "steam_games_dataset.db")
