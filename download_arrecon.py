"""Download QC dropsonde files from AR Recon (cw3e-datashare.ucsd.edu/ARRecon/).

Structure: ARRecon/YEAR/IOP/AIRCRAFT/files
Downloads only files with 'QC' in the name.
Skips files that already exist locally.
"""
import re
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://cw3e-datashare.ucsd.edu/ARRecon"
LOCAL_BASE = Path("data/arrecon")
MAX_WORKERS = 1


def list_dir(url):
    """Return relative hrefs from an Apache directory listing, skipping query strings."""
    r = requests.get(url + "/", timeout=30)
    r.raise_for_status()
    hrefs = re.findall(r'href="([^"?][^"]*)"', r.text)
    # Keep only relative links (not absolute, not style.css, not parent)
    return [h for h in hrefs if not h.startswith("/") and not h.startswith("http")]


def crawl():
    """Walk YEAR/IOP/AIRCRAFT and return (url, local_path) for every QC file."""
    tasks = []
    years = [h.rstrip("/") for h in list_dir(BASE_URL) if re.match(r"\d{4}/", h)]
    for year in years:
        iops = [h.rstrip("/") for h in list_dir(f"{BASE_URL}/{year}") if h.endswith("/")]
        for iop in iops:
            aircrafts = [h.rstrip("/") for h in list_dir(f"{BASE_URL}/{year}/{iop}") if h.endswith("/")]
            for ac in aircrafts:
                files = list_dir(f"{BASE_URL}/{year}/{iop}/{ac}")
                for f in files:
                    if "QC" in f and not f.endswith("/"):
                        url = f"{BASE_URL}/{year}/{iop}/{ac}/{f}"
                        local = LOCAL_BASE / year / iop / ac / f
                        tasks.append((url, local))
    return tasks


def download(url, local_path):
    if local_path.exists():
        return f"skip  {local_path.relative_to(LOCAL_BASE)}"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    local_path.write_bytes(r.content)
    return f"done  {local_path.relative_to(LOCAL_BASE)}"


def main():
    print("Crawling directory structure...", flush=True)
    tasks = crawl()
    print(f"Found {len(tasks)} QC files to download", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download, url, path): (url, path) for url, path in tasks}
        for fut in as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as e:
                url, path = futures[fut]
                print(f"ERROR {url}: {e}", flush=True)

    print("All done.")


if __name__ == "__main__":
    main()
