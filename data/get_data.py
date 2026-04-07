import pandas as pd

df = pd.read_parquet("hf://datasets/FronkonGames/steam-games-dataset/data/train-00000-of-00001.parquet")

# export df to csv
df.to_csv("steam_games_dataset.csv", index=False)