# CLAUDE.md — sonde-regrid

## Testing

All tests in this repo validate the **generated output datasets** (NetCDF files in `output/`).
Running tests without first regenerating the datasets after code changes is not useful — the
tests will pass or fail based on stale data.

**Always confirm with the user before regenerating datasets.** Regeneration reads large source
data files and can take significant time.

## Project layout

- `src/` — readers, regridding, diagnostics, and the processing pipeline (`process.py`)
- `tests/` — validation tests that run against files in `output/`
- `doc/` — LaTeX spec (`regridding.tex`)
- `data/` — source datasets (not tracked in git)
- `output/` — generated NetCDF files (not tracked in git)
