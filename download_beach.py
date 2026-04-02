"""Download BEACH (ORCESTRA) dropsonde Level_2 and Level_3 zarr stores."""
import requests
import re
import xarray as xr
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://latest.orcestra-campaign.org/products/HALO/dropsondes"
LOCAL_BASE = Path("data/beach")
LEVELS = ["Level_2", "Level_3"]
MAX_WORKERS = 8


TIMEOUT = 120  # seconds for HTTP requests and zarr chunk fetches


def get_flight_days(level):
    url = f"{BASE_URL}/{level}/"
    r = requests.get(url, timeout=TIMEOUT)
    return re.findall(rf'href="/products/HALO/dropsondes/{level}/(HALO-[^"]+)"', r.text)


def get_zarr_names(level, flight):
    url = f"{BASE_URL}/{level}/{flight}/"
    r = requests.get(url, timeout=TIMEOUT)
    return re.findall(
        rf'href="/products/HALO/dropsondes/{level}/{flight}/([^"]+\.zarr)"', r.text
    )


def is_complete(dst):
    """A zarr store is complete if it has a .complete sentinel written after download."""
    return (dst / ".complete").exists()


def download_store(level, flight, name):
    dst = LOCAL_BASE / level / flight / name
    if is_complete(dst):
        return f"skip  {level}/{flight}/{name}"
    src_url = f"{BASE_URL}/{level}/{flight}/{name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    storage_options = {"timeout": TIMEOUT}
    ds = xr.open_dataset(src_url, engine="zarr", storage_options=storage_options)
    ds.load().to_zarr(dst, mode="w")
    (dst / ".complete").touch()
    return f"done  {level}/{flight}/{name}"


def get_top_level_zarrs(level):
    """List zarr stores directly under a level (no flight subdirectories)."""
    url = f"{BASE_URL}/{level}/"
    r = requests.get(url, timeout=TIMEOUT)
    return re.findall(
        rf'href="/products/HALO/dropsondes/{level}/([^"]+\.zarr)"', r.text
    )


def download_top_level_store(level, name):
    """Download a zarr store that sits directly under a level (e.g. Level_3)."""
    dst = LOCAL_BASE / level / name
    if is_complete(dst):
        return f"skip  {level}/{name}"
    src_url = f"{BASE_URL}/{level}/{name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    storage_options = {"timeout": TIMEOUT}
    ds = xr.open_dataset(src_url, engine="zarr", storage_options=storage_options)
    ds.load().to_zarr(dst, mode="w")
    (dst / ".complete").touch()
    return f"done  {level}/{name}"


def main():
    tasks = []          # (func, args) tuples
    for level in LEVELS:
        # First try per-flight layout (Level_2)
        try:
            flights = get_flight_days(level)
        except Exception as e:
            flights = []
            print(f"Could not list flight dirs for {level}: {e}")

        if flights:
            for flight in flights:
                try:
                    names = get_zarr_names(level, flight)
                except Exception as e:
                    print(f"Could not list {level}/{flight}: {e}")
                    continue
                for name in names:
                    tasks.append((download_store, (level, flight, name)))
        else:
            # Fall back to top-level zarr stores (Level_3)
            try:
                names = get_top_level_zarrs(level)
            except Exception as e:
                print(f"Could not list top-level zarrs for {level}: {e}")
                continue
            for name in names:
                tasks.append((download_top_level_store, (level, name)))

    print(f"Found {len(tasks)} stores across {LEVELS}")

    done = 0
    total = len(tasks)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(func, *args): args for func, args in tasks}
        for fut in as_completed(futures):
            done += 1
            try:
                print(f"[{done}/{total}] {fut.result()}", flush=True)
            except Exception as e:
                print(f"[{done}/{total}] ERROR {futures[fut]}: {e}", flush=True)

    print("All done.")


if __name__ == "__main__":
    main()
