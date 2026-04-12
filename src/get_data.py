import pandas as pd

from data_io import write_data

if __name__ == "__main__":
    df = pd.read_parquet(
        "hf://datasets/FronkonGames/steam-games-dataset/data/train-00000-of-00001.parquet"
    )

    write_data(df, "steam_games_dataset.db")
