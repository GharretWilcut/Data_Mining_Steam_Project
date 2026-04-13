# Run data preprocessing step

import json
from datetime import datetime

import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from data_io import read_data, write_data

ss = StandardScaler()
mlb: dict[str, MultiLabelBinarizer] = {}

OWNER_COUNT_RANGES = [
    "0 - 0",
    "0 - 20000",
    "20000 - 50000",
    "50000 - 100000",
    "100000 - 200000",
    "200000 - 500000",
    "500000 - 1000000",
    "1000000 - 2000000",
    "2000000 - 5000000",
    "5000000 - 10000000",
    "10000000 - 20000000",
    "20000000 - 50000000",
    "50000000 - 100000000",
    "100000000 - 200000000",
]


def preprocess(df: pd.DataFrame, fit: bool):
    clean_df = pd.DataFrame()

    # Direct features
    clean_df["price"] = df["price"]
    clean_df["dlc_count"] = df["dlc_count"]

    # Date when dataset was created, used for age calculation
    reference_date = datetime(2026, 2, 2)
    # Extract date features
    dates = df["release_date"].map(lambda x: datetime.strptime(x, "%b %d, %Y"))
    clean_df["day_of_year"] = dates.map(lambda x: x.timetuple().tm_yday / 366)
    clean_df["age_in_days"] = dates.map(lambda x: (reference_date - x).days)

    # Label-encode estimated_owners and move to y
    y = df["estimated_owners"].map(lambda x: OWNER_COUNT_RANGES.index(x))

    # Boolean casted features
    clean_df["has_achievements"] = df["achievements"].map(lambda x: 0 if x == 0 else 1)
    clean_df["has_website"] = df["website"].map(lambda x: 0 if x == "" else 1)
    clean_df["has_support_url"] = df["support_url"].map(lambda x: 0 if x == "" else 1)
    clean_df["has_support_email"] = df["support_email"].map(
        lambda x: 0 if x == "" else 1
    )

    # Platform int casts
    clean_df["windows"] = df["windows"].map(int)
    clean_df["mac"] = df["mac"].map(int)
    clean_df["linux"] = df["linux"].map(int)

    # One-hot encoding of list columns
    list_columns = [
        "genres",
        "categories",
        "supported_languages",
        "full_audio_languages",
    ]
    for column in list_columns:
        column_contents = df[column].map(lambda x: x.split(","))
        if fit:
            mlb[column] = MultiLabelBinarizer()
            encoded = mlb[column].fit_transform(column_contents)
        else:
            encoded = mlb[column].transform(column_contents)
        labels = pd.DataFrame(
            encoded,
            columns=mlb[column].classes_,
            index=df.index,
        )
        labels = labels.drop(columns=[""]).add_prefix(f"{column} ")
        clean_df = clean_df.join(labels)

    # Label encoding of required age
    # 0-10 -> E (0)
    # 11-15 -> T (1)
    # 16+ -> M (2)
    clean_df["required_age"] = df["required_age"].map(
        lambda x: (0 if x <= 10 else 1 if x <= 15 else 2) / 2
    )

    # Count number of packages
    clean_df["package_count"] = df["packages"].map(lambda x: len(json.loads(x)))

    standard_columns = ["price", "dlc_count", "age_in_days", "package_count"]
    if fit:
        clean_df[standard_columns] = ss.fit_transform(clean_df[standard_columns])
    else:
        clean_df[standard_columns] = ss.transform(clean_df[standard_columns])

    # TODO text embeddings/keyword extraction
    text_columns = ["name", "short_description", "detailed_description", "notes"]

    # TODO fetch from links + autoencoder
    image_columns = ["header_image", "screenshots"]

    # TODO Feature selection (e.g., variance threshold, model-based selection, PCA)

    return clean_df, y


if __name__ == "__main__":
    df_full = read_data("steam_games_dataset.db")

    print("Data loaded")

    # Hardcoded name imputation
    df_full.at[44432, "name"] = "The Spookening"

    df_train: pd.DataFrame
    df_test: pd.DataFrame
    df_train, df_test = model_selection.train_test_split(
        df_full, test_size=0.25, random_state=42
    )

    df_train_processed, y_train = preprocess(df_train, True)
    df_test_processed, y_test = preprocess(df_test, False)

    df_train_processed["estimated_owners"] = y_train
    df_test_processed["estimated_owners"] = y_test

    write_data(df_train_processed, "steam_games_dataset_clean_training.db")
    write_data(df_test_processed, "steam_games_dataset_clean_testing.db")

    print("Data written")
