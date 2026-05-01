# Steam Image Downloader
# Downloads Steam game images into data/images/
# Filenames are saved as {appid}.jpg to match extract_embeddings.py and integrate_embeddings.py

import os
import time
import requests

""" 
Go to https://steamcommunity.com/dev/apikey
Sign in and set domain name to 'localhost'
Copy the key and paste it below
You might just be able to use this key, but I'm not sure
"""
STEAM_API_KEY = "8A6DC3DB995E263AC32BC5049AF00CC6"

MAX_GAMES = 5000 # Set to integer for testing, set to None to download all

OUTPUT_DIR = "data/images"
DELAY = 0.2 


def get_steam_app_list():
    print("Fetching Steam app list...")

    if STEAM_API_KEY:
        url = (
            f"https://api.steampowered.com/IStoreService/GetAppList/v1/"
            f"?key={STEAM_API_KEY}&include_games=1&max_results=50000"
        )
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                apps = response.json()["response"]["apps"]
                print(f"Found {len(apps)} apps via Steam API key.")
                return [{"appid": a["appid"]} for a in apps]
        except Exception as e:
            print(f"  API key request failed: {e} — falling back to GitHub list...")



def filter_games_only(apps, max_games=None):
    # Trim list to max_games if set.
    if max_games:
        apps = apps[:max_games]
    return apps


def download_images(apps, output_dir, delay=0.2):
    os.makedirs(output_dir, exist_ok=True)

    success = 0
    skipped = 0
    failed = 0

    for app in apps:
        appid = app["appid"]
        out_path = os.path.join(output_dir, f"{appid}.jpg")

        # Skip already downloaded
        if os.path.exists(out_path):
            skipped += 1
            continue

        url = f"https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg"

        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
                success += 1
                if success % 100 == 0:
                    print(f"  Downloaded {success} images...")
            else:
                failed += 1  # 404 = no image
        except Exception as e:
            print(f"  Error on {appid}: {e}")
            failed += 1

        time.sleep(delay)

    print(f"\nDone. Success: {success} | Skipped (already existed): {skipped} | Failed/missing: {failed}")
    print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    apps = get_steam_app_list()
    apps = filter_games_only(apps, max_games=MAX_GAMES)
    download_images(apps, output_dir=OUTPUT_DIR, delay=DELAY)