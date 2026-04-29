import pandas as pd

DATASET = "data/games.csv" # This is the dataset to be merged with. Change name as needed.
EMBEDDINGS = "data/embeddings.csv"

# Load datasets
games = pd.read_csv(DATASET)
embeddings = pd.read_csv(EMBEDDINGS)

# Merge on game_id
merged = games.merge(embeddings, on="game_id")

# Basic checks
print("Games shape:", games.shape)
print("Embeddings shape:", embeddings.shape)
print("Merged shape:", merged.shape)

# Check for missing matches
missing = games.shape[0] - merged.shape[0]
print(f"Missing embeddings for {missing} games")

# Save result
merged.to_csv("data/final_dataset.csv", index=False)

print("Saved merged dataset to final_dataset.csv")