# Run data preprocessing step

import pandas as pd

from data_io import read_data, write_data

"""
Game Format

{
    "appID": int
    "name": string
    "release_date": string
    "estimated_owners": int
    "peak_ccu": int
    "required_age": int
    "price": float
    "dlc_count": int
    "detailed_description": string
    "short_description": string
    "supported_languages": string[]
    "full_audio_languages": string[]
    "reviews": string
    "header_image": string
    "website": string
    "support_url": string
    "support_email": string
    "windows": bool
    "mac": bool
    "linux": bool
    "metacritic_score": int
    "metacritic_url": string
    "user_score": int
    "positive": int
    "negative": int
    "score_rank": string
    "achievements": int
    "recommendations": int
    "notes": string
    "average_playtime_forever": int
    "average_playtime_2weeks": int
    "median_playtime_forever": int
    "median_playtime_2weeks": int
    "developers": string[]
    "publishers": string[]
    "categories": string[]
    "genres": string[]
    "tags": string[]
    "screenshots": string[]
    "movies": string[]
    "packages": {
        "title": string
        "description": string
        "subs": {
            "text": string
            "description": string
            "price": float
        }[]
    }[]
}
"""

if __name__ == "__main__":
    df = read_data("steam_games_dataset.db")

    # Hardcoded name imputation
    df.at[44432, "name"] = "The Spookening"

    # TODO
    # Train / Test Split
    # Missing Value Handling
    # Outlier Handling
    # Feature Construction
    # Categorical Encoding
    # Feature Scaling
    # Feature Selection
    clean_df = pd.DataFrame(df)

    write_data(clean_df, "steam_games_dataset_clean.db")
