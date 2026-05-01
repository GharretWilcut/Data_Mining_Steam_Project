
import asyncio
import os
import sqlite3

import aiofiles
import aiohttp
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────
DB_PATH     = "data/steam_games_dataset.db"
OUTPUT_DIR  = "data/images"
LOG_PATH    = os.path.join(OUTPUT_DIR, "download_log.csv")
CONCURRENCY = 50    # concurrent requests — raise to 100 if your connection allows
TIMEOUT     = 15    # seconds per request
# ─────────────────────────────────────────────────────────────────────────────


def load_urls_from_db(db_path: str) -> list[tuple[int, str]]:
    """
    Read (appID, header_image) from the games table.
    Skips rows with empty/null URLs.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT appID, header_image FROM games WHERE header_image IS NOT NULL AND header_image != ''",
        conn
    )
    conn.close()

    pairs = list(zip(df["appID"].astype(int), df["header_image"].astype(str)))
    print(f"Found {len(pairs):,} games with a header_image URL in the DB.")
    return pairs


def load_already_done(log_path: str) -> set[int]:
    if not os.path.exists(log_path):
        return set()
    log = pd.read_csv(log_path)
    return set(log[log["status"] == "ok"]["appid"].astype(int).tolist())


def flush_log(log_path: str, rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(log_path)
    df.to_csv(log_path, mode="a", index=False, header=header)


async def download_one(
    session: aiohttp.ClientSession,
    appid: int,
    url: str,
    output_dir: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    out_path = os.path.join(output_dir, f"{appid}.jpg")
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
                if r.status == 200:
                    content = await r.read()
                    async with aiofiles.open(out_path, "wb") as f:
                        await f.write(content)
                    return {"appid": appid, "status": "ok"}
                else:
                    return {"appid": appid, "status": f"http_{r.status}"}
        except asyncio.TimeoutError:
            return {"appid": appid, "status": "timeout"}
        except Exception as e:
            return {"appid": appid, "status": f"error: {e}"}


async def download_all(pairs: list[tuple[int, str]], output_dir: str, log_path: str):
    os.makedirs(output_dir, exist_ok=True)

    already_done = load_already_done(log_path)
    todo = [(appid, url) for appid, url in pairs if appid not in already_done]

    print(f"{len(already_done):,} already done, {len(todo):,} remaining.\n")

    if not todo:
        print("Nothing to do.")
        return

    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)

    success = 0
    failed  = 0
    log_buffer: list[dict] = []
    FLUSH_EVERY = 200

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            download_one(session, appid, url, output_dir, semaphore)
            for appid, url in todo
        ]

        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            result = await coro
            log_buffer.append(result)

            if result["status"] == "ok":
                success += 1
            else:
                failed += 1

            if len(log_buffer) >= FLUSH_EVERY:
                flush_log(log_path, log_buffer)
                log_buffer = []

            if i % 1000 == 0:
                pct = i / len(todo) * 100
                print(f"  [{i:,}/{len(todo):,}] {pct:.1f}%  ✓ {success:,}  ✗ {failed:,}")

    flush_log(log_path, log_buffer)

    print(f"\n── Done ─────────────────────────────────────────────")
    print(f"  ✓ {success:,} downloaded")
    print(f"  ✗ {failed:,} failed / missing")
    print(f"  Log    : {log_path}")
    print(f"  Images : {output_dir}/")


if __name__ == "__main__":
    pairs = load_urls_from_db(DB_PATH)
    asyncio.run(download_all(pairs, OUTPUT_DIR, LOG_PATH))