"""Clean up ARRecon download: keep .frd only, plus .nc files that have no .frd counterpart.

Run with DRY_RUN = True first to verify, then set to False to delete.
"""
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data/arrecon")
DRY_RUN = False  # set to False to actually delete

# Extensions to always delete (redundant formats)
DELETE_EXTENSIONS = {".bfr", ".csv", ".cls", ".eol",
                     ".frd_bad", ".frd_baddrop", ".frd_noPTH", ".frd_noWinds", ".txt"}

# Build a set of stems that have a .frd file (strip full suffix e.g. "QC.frd" → stem)
def stem_of(f):
    suffix = "".join(f.suffixes)
    return f.parent / f.name.replace(suffix, "")

frd_stems = {stem_of(f) for f in DATA_DIR.rglob("*")
             if "".join(f.suffixes) == ".frd"}

to_delete = []
to_keep = []

for f in sorted(DATA_DIR.rglob("*")):
    if not f.is_file():
        continue
    if f.name == "index.html":
        to_delete.append(f)
        continue

    suffix = "".join(f.suffixes)
    stem = stem_of(f)

    if suffix in DELETE_EXTENSIONS:
        to_delete.append(f)
    elif suffix == ".nc":
        # Keep only if no .frd counterpart exists
        if stem in frd_stems:
            to_delete.append(f)
        else:
            to_keep.append(f)
    elif suffix == ".frd":
        to_keep.append(f)
    else:
        # Unknown extension — report but don't delete
        print(f"UNKNOWN: {f.relative_to(DATA_DIR)}")

# Summary
by_ext = defaultdict(int)
for f in to_delete:
    by_ext["".join(f.suffixes)] += 1

print(f"\nWill DELETE {len(to_delete)} files:")
for ext, n in sorted(by_ext.items()):
    print(f"  {ext or '(no ext)'}: {n}")
print(f"\nWill KEEP {len(to_keep)} files ({sum(1 for f in to_keep if f.suffix == '.frd')} frd, "
      f"{sum(1 for f in to_keep if f.suffix == '.nc')} nc-only)")

if DRY_RUN:
    print("\nDRY RUN — set DRY_RUN = False to actually delete.")
else:
    for f in to_delete:
        f.unlink()
    print("\nDone.")
