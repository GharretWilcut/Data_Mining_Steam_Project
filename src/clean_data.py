# Run data preprocessing step

from data_io import read_data, write_data

"""
Game Format

{
    "name": string
    "release_date": string
    "required_age": int
    "price": float
    "dlc_count": int
    "detailed_description": string
    "short_description": string
    "reviews": string
    "header_image": string
    "website": string
    "support_url": string
    "support_email": string
    "windows": bool
    "mac": bool
    "linux": bool
    "metacritic_score": float
    "metacritic_url": string
    "achievements": int
    "recommendations": int
    "notes": string
    "supported_languages": string[]
    "full_audio_languages": string[]
    "packages": {
        "title": string
        "description": string
        "subs": {
            "text": string
            "description": string
            "price": float
        }[]
    }[]
    "developers": string[]
    "publishers": string[]
    "categories": string[]
    "genres": string[]
    "screenshots": string[]
    "movies": string[]
}
"""

if __name__ == "__main__":
    df = read_data()

    missing = {
        "name": df[df["name"] == ""].shape[0],
        "release_date": df[df["release_date"] == ""].shape[0],
        "required_age": df[df["required_age"] == 0].shape[0],
        "price": df[df["price"] == 0].shape[0],
        "dlc_count": df[df["dlc_count"] == 0].shape[0],
        "detailed_description": df[df["detailed_description"] == ""].shape[0],
        "short_description": df[df["short_description"] == ""].shape[0],
        "reviews": df[df["reviews"] == ""].shape[0],
        "header_image": df[df["header_image"] == ""].shape[0],
        "website": df[df["website"] == ""].shape[0],
        "support_url": df[df["support_url"] == ""].shape[0],
        "support_email": df[df["support_email"] == ""].shape[0],
        "metacritic_score": df[df["metacritic_score"] == ""].shape[0],
        "metacritic_url": df[df["metacritic_url"] == ""].shape[0],
        "achievements": df[df["achievements"] == 0].shape[0],
        "recommendations": df[df["recommendations"] == 0].shape[0],
        "notes": df[df["notes"] == ""].shape[0],
        "supported_languages": df[df["supported_languages"] == b""].shape[0],
        "full_audio_languages": df[df["full_audio_languages"] == b""].shape[0],
        "packages": df[df["packages"] == b""].shape[0],
        "developers": df[df["developers"] == b""].shape[0],
        "publishers": df[df["publishers"] == b""].shape[0],
        "categories": df[df["categories"] == b""].shape[0],
        "genres": df[df["genres"] == b""].shape[0],
        "screenshots": df[df["screenshots"] == b""].shape[0],
        "movies": df[df["movies"] == b""].shape[0],
    }

    print("Missing value counts:", missing)

    # Hardcoded name fix
    df.at[44432, "name"] = "The Spookening"

    write_data(df)
