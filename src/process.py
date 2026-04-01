"""
Process all datasets: read → regrid → diagnose → write NetCDF.

Produces one NetCDF per dropsonde dataset and one per year for IGRA.
Output dimensions are (launch_location, launch_time, altitude).
For dropsondes, launch_time has size 1 (one sounding per location).
For radiosondes, profiles are grouped by station.

Usage:
    python process.py                 # process all datasets
    python process.py joanne otrec    # process specific datasets
"""

import os
import sys
import time
from collections import defaultdict

import numpy as np
import xarray as xr

from regrid import regrid_sonde, make_grid, DEFAULT_DZ
from readers import (
    read_joanne,
    read_beach,
    read_otrec,
    read_activate,
    read_hurricane,
    read_dynamo,
    read_shout,
    read_arrecon,
    read_haloac3,
    read_enrr,
    read_hs3,
    read_predict,
    read_igra,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "output")

Z_MIN = 0.0
Z_MAX_DEFAULT = 20000.0   # 20 km for dropsonde datasets
Z_MAX_IGRA = 40000.0      # 40 km for IGRA radiosondes
DZ = DEFAULT_DZ

IGRA_YEAR_MIN = 2025
IGRA_YEAR_MAX = 2025
IGRA_SUBSAMPLE = 10       # keep every Nth day (1 = all, 10 = DOY 1, 11, 21, ...)

DATA_VARIABLES = ["u", "v", "p", "T", "RH", "q", "theta", "theta_e", "MSE", "DSE"]

VARIABLE_ATTRS = {
    "u":       {"units": "m s-1",   "long_name": "zonal wind",
                "standard_name": "eastward_wind"},
    "v":       {"units": "m s-1",   "long_name": "meridional wind",
                "standard_name": "northward_wind"},
    "p":       {"units": "Pa",      "long_name": "pressure",
                "standard_name": "air_pressure"},
    "T":       {"units": "K",       "long_name": "temperature",
                "standard_name": "air_temperature"},
    "RH":      {"units": "%",       "long_name": "relative humidity",
                "standard_name": "relative_humidity"},
    "q":       {"units": "kg kg-1", "long_name": "water vapor mixing ratio",
                "standard_name": "humidity_mixing_ratio"},
    "theta":   {"units": "K",       "long_name": "potential temperature",
                "standard_name": "air_potential_temperature"},
    "theta_e": {"units": "K",       "long_name": "equivalent potential temperature",
                "standard_name": "equivalent_potential_temperature",
                "comment": "Bolton (1980) formulation"},
    "MSE":     {"units": "J kg-1",  "long_name": "moist static energy",
                "comment": "cp*T + g*z + Lv*q"},
    "DSE":     {"units": "J kg-1",  "long_name": "dry static energy",
                "comment": "cp*T + g*z"},
}

DATASETS = {
    "joanne": {
        "reader": read_joanne,
        "path": os.path.join(DATA_DIR, "joanne"),
        "z_max": Z_MAX_DEFAULT,
        "provenance": {
            "campaign": "EUREC4A",
            "instrument": "Vaisala RD41 dropsonde",
            "platform": "HALO, NOAA WP-3D",
            "region": "Tropical North Atlantic, east of Barbados",
            "period": "January-February 2020",
            "citation": "George et al. (2021), doi:10.5194/essd-13-5253-2021",
            "qc_applied": "Only profiles flagged 'good' are used (1068 of 1215). "
                          "RH from HALO sondes corrected by factor 1.06 in source data.",
            "source_product": "JOANNE Level 2 individual sounding profiles",
        },
    },
    "beach": {
        "reader": read_beach,
        "path": os.path.join(DATA_DIR, "beach"),
        "z_max": Z_MAX_DEFAULT,
        "provenance": {
            "campaign": "ORCESTRA (PERCUSION/MAESTRO)",
            "instrument": "Vaisala RD41 dropsonde",
            "platform": "HALO",
            "region": "Eastern and western tropical Atlantic",
            "period": "August-September 2024",
            "citation": "Gloeckner et al. (2025), doi:10.5194/essd-2025-22",
            "qc_applied": "Only profiles with sonde_qc=0 (GOOD) are used (976 of 1191). "
                          "17 misconfigured minisondes and 1 factory-reset sonde (0bd0e322) "
                          "excluded via sonde_qc flag.",
            "source_product": "BEACH Level 2 individual sounding profiles (Zarr)",
        },
    },
    "otrec": {
        "reader": read_otrec,
        "path": os.path.join(DATA_DIR, "otrec"),
        "z_max": Z_MAX_DEFAULT,
        "provenance": {
            "campaign": "OTREC",
            "instrument": "NCAR NRD41 dropsonde",
            "platform": "NSF/NCAR Gulfstream-V",
            "region": "Tropical eastern Pacific and Caribbean",
            "period": "August-October 2019",
            "citation": "Vomel et al. (2021), doi:10.26023/GSWK-AQ13-4G09",
            "qc_applied": "ASPEN QC with manual post-processing. Per-sonde pressure offset "
                          "and RH time-lag corrections applied in source data. "
                          "One warm-biased sounding (20190925_154412) excluded.",
            "source_product": "OTREC AVAPS NRD41 v1 NetCDFs",
        },
    },
    "activate": {
        "reader": read_activate,
        "path": os.path.join(DATA_DIR, "activate"),
        "z_max": Z_MAX_DEFAULT,
        "provenance": {
            "campaign": "ACTIVATE",
            "instrument": "Vaisala dropsonde",
            "platform": "NASA King Air",
            "region": "Western North Atlantic, off eastern US coast and Bermuda",
            "period": "February 2020 - June 2022 (6 deployment periods)",
            "citation": "Vomel et al. (2023), doi:10.1038/s41597-023-02647-5",
            "qc_applied": "Only 'Good Drop' profiles retained. Corrected launch times used "
                          "for 17 sondes with synchronization errors. RH excluded for "
                          "20 sondes with unreconditioned humidity sensors.",
            "source_product": "ACTIVATE ICT dropsonde files",
        },
    },
    "hurricane": {
        "reader": read_hurricane,
        "path": os.path.join(DATA_DIR, "hurricane",
                             "1996-2012.NOAA.Hurricane.Dropsonde.Archive.v2"),
        "z_max": Z_MAX_DEFAULT,
        "provenance": {
            "campaign": "NOAA Hurricane Reconnaissance/Surveillance",
            "instrument": "GPS dropsonde (AVAPS/AVAPS II)",
            "platform": "NOAA Gulfstream-IV, NOAA WP-3D",
            "region": "Atlantic and eastern Pacific tropical cyclones",
            "period": "1996-2012 (120 tropical cyclones)",
            "citation": "Wang et al. (2015), doi:10.1175/MWR-D-14-00290.1",
            "qc_applied": "ASPEN QC with visual inspection. Archive used as provided.",
            "source_product": "NOAA Hurricane Dropsonde Archive v2 (EOL format)",
        },
    },
    "dynamo": {
        "reader": read_dynamo,
        "path": os.path.join(DATA_DIR, "dynamo"),
        "z_max": Z_MAX_DEFAULT,
        "provenance": {
            "campaign": "DYNAMO",
            "instrument": "Vaisala RD94 dropsonde",
            "platform": "NOAA P-3",
            "region": "Central Indian Ocean (14.5S-1.2N, 69.8-80.9E)",
            "period": "November-December 2011 (13 research flights)",
            "citation": "Ciesielski et al. (2014), doi:10.1175/BAMS-D-13-00016.1",
            "qc_applied": "ASPEN QC. Version 3 release with temperature-dependent "
                          "RH dry bias correction for RD94 dropsondes.",
            "source_product": "DYNAMO dropsonde archive (NetCDF)",
        },
    },
    "shout": {
        "reader": read_shout,
        "path": os.path.join(DATA_DIR, "shout"),
        "z_max": Z_MAX_DEFAULT,
        "provenance": {
            "campaign": "SHOUT",
            "instrument": "GPS dropsonde",
            "platform": "NASA Global Hawk (cruise altitude ~18 km)",
            "region": "Atlantic tropical cyclones, Pacific winter storms",
            "period": "2015-2016 (15 missions, 638 sondes)",
            "citation": "Wick et al. (2020), doi:10.1175/BAMS-D-18-0279.1",
            "qc_applied": "ASPEN QC. Archive used as provided.",
            "source_product": "SHOUT dropsonde archive (EOL format)",
        },
    },
    "arrecon": {
        "reader": read_arrecon,
        "path": os.path.join(DATA_DIR, "arrecon"),
        "z_max": Z_MAX_DEFAULT,
        "provenance": {
            "campaign": "Atmospheric River Reconnaissance (AR Recon)",
            "instrument": "GPS dropsonde",
            "platform": "NOAA G-IVSP, Air Force WC-130J",
            "region": "Eastern North Pacific, approaching US West Coast",
            "period": "2018-2026 winter seasons (November-March)",
            "citation": "Cobb et al. (2022), doi:10.1175/WAF-D-21-0104.1",
            "qc_applied": "ASPEN QC. Archive used as provided.",
            "source_product": "AR Recon FRD dropsonde files",
        },
    },
    "haloac3": {
        "reader": read_haloac3,
        "path": os.path.join(DATA_DIR, "halo-ac3"),
        "z_max": Z_MAX_DEFAULT,
        "provenance": {
            "campaign": "HALO-(AC)³",
            "instrument": "Vaisala RD41 dropsonde",
            "platform": "HALO, Polar 5",
            "region": "Norwegian and Greenland Seas, Fram Strait, central Arctic",
            "period": "March-April 2022",
            "citation": "Ehrlich et al. (2025), doi:10.5194/essd-17-1295-2025",
            "qc_applied": "Quality control and processing applied to Level-1 data. "
                          "George et al. (2024), doi:10.1594/PANGAEA.968891",
            "source_product": "HALO-(AC)³ Level 2 individual sounding profiles (NetCDF)",
        },
    },
    "enrr": {
        "reader": read_enrr,
        "path": os.path.join(DATA_DIR, "enrr"),
        "z_max": Z_MAX_DEFAULT,
        "provenance": {
            "campaign": "NOAA El Niño Rapid Response (ENRR)",
            "instrument": "GPS dropsonde",
            "platform": "NOAA G-IV, NOAA C-130, NASA Global Hawk",
            "region": "Tropical Pacific, Hawaii to U.S. West Coast",
            "period": "January-March 2016",
            "citation": "Dole et al. (2018), doi:10.1175/BAMS-D-16-0219.1",
            "qc_applied": "ASPEN QC. G-IV and Global Hawk archives use corrected data. "
                          "C-130 archive applies temperature-dependent dry bias correction.",
            "source_product": "ENRR corrected dropsonde archives (FRD and EOL format)",
        },
    },
    "hs3": {
        "reader": read_hs3,
        "path": os.path.join(DATA_DIR, "hs3"),
        "z_max": Z_MAX_DEFAULT,
        "provenance": {
            "campaign": "NASA Hurricane and Severe Storm Sentinel (HS3)",
            "instrument": "GPS dropsonde",
            "platform": "NASA Global Hawk (cruise altitude ~18 km)",
            "region": "Atlantic tropical cyclones",
            "period": "2011-2014 hurricane seasons",
            "citation": "Braun et al. (2016), doi:10.1175/BAMS-D-15-00186.1",
            "qc_applied": "ASPEN QC. Version 3 release.",
            "source_product": "HS3 dropsonde archive (EOL format)",
        },
    },
    "predict": {
        "reader": read_predict,
        "path": os.path.join(DATA_DIR, "predict"),
        "z_max": Z_MAX_DEFAULT,
        "provenance": {
            "campaign": "PREDICT",
            "instrument": "EOL/Vaisala GPS dropsonde",
            "platform": "NSF/NCAR Gulfstream-V",
            "region": "Caribbean and western Atlantic",
            "period": "August-September 2010",
            "citation": "Montgomery et al. (2012), doi:10.1175/BAMS-D-11-00046.1",
            "qc_applied": "ASPEN QC. Version 3 release with temperature-dependent "
                          "RH and dewpoint dry bias correction.",
            "source_product": "PREDICT dropsonde archive (EOL format)",
        },
    },
    "igra": {
        "reader": read_igra,
        "path": os.path.join(DATA_DIR, "igra"),
        "z_max": Z_MAX_IGRA,
        "provenance": {
            "campaign": "IGRA (Integrated Global Radiosonde Archive)",
            "instrument": "Various radiosondes",
            "platform": "Ground stations (1500+ globally)",
            "region": "Global",
            "period": "2025 (subsampled every 10th day)",
            "citation": "Durre et al. (2006), doi:10.1175/JCLI3594.1",
            "qc_applied": "IGRA automated QA (physical plausibility, internal consistency, "
                          "climatological outliers, temporal consistency). "
                          "Geopotential height converted to geometric altitude.",
            "source_product": "IGRA v2 station data files",
        },
    },
}


def _regrid_profile(prof, z_max):
    """Regrid a single profile dict, returning a dict of 1-D arrays."""
    variables = {}
    for var in ("u", "v", "p", "T", "RH"):
        if var in prof and prof[var] is not None:
            variables[var] = prof[var]
    if "q" in prof and prof["q"] is not None:
        variables["q"] = prof["q"]
    obs_time = prof.get("obs_time")
    ds = regrid_sonde(prof["altitude"], variables, z_min=Z_MIN, z_max=z_max, dz=DZ,
                      obs_time=obs_time)
    result = {var: ds[var].values for var in DATA_VARIABLES if var in ds}
    if "observation_time" in ds:
        result["observation_time"] = ds["observation_time"].values
    return result


ALTITUDE_ATTRS = {
    "units": "m", "long_name": "altitude above mean sea level",
    "axis": "Z", "positive": "up",
}
LAUNCH_LAT_ATTRS = {"units": "degrees_north", "standard_name": "latitude"}
LAUNCH_LON_ATTRS = {"units": "degrees_east", "standard_name": "longitude"}


def _set_coord_attrs(out, lat_long_name="latitude at profile start",
                     lon_long_name="longitude at profile start"):
    """Attach standard attributes to coordinates and data variables."""
    out["altitude"].attrs = ALTITUDE_ATTRS
    out["launch_lat"].attrs = {**LAUNCH_LAT_ATTRS, "long_name": lat_long_name}
    out["launch_lon"].attrs = {**LAUNCH_LON_ATTRS, "long_name": lon_long_name}
    out["launch_time"].attrs = {"long_name": "date and time of sounding launch"}
    out["observation_time"].attrs = {"long_name": "observation time at each altitude level"}
    for var, attrs in VARIABLE_ATTRS.items():
        if var in out:
            out[var].attrs = attrs


def _global_attrs(name, z_max, n_locations, n_soundings, n_alt, provenance=None):
    """Standard global attributes with per-dataset provenance."""
    attrs = {
        "title": f"Regridded sonde profiles — {name}",
        "source": f"Bin-averaged from {name} dataset",
        "history": f"Created by sonde-regrid/src/process.py on "
                   f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "Conventions": "CF-1.8",
        "regridding_method": "bin averaging (no interpolation)",
        "grid_spacing_m": DZ,
        "grid_min_m": Z_MIN,
        "grid_max_m": z_max,
        "n_launch_locations": n_locations,
        "n_soundings": n_soundings,
        "n_altitude_levels": n_alt,
        "variables_directly_gridded": "u, v, p, T, RH",
        "variables_diagnosed_post_gridding": "q, theta, theta_e, MSE, DSE",
        "missing_value_note": "NaN indicates grid cells with no valid observations",
    }
    if provenance:
        for key, val in provenance.items():
            attrs[f"source_{key}"] = val
    return attrs


def process_dataset(name, reader, data_path, z_max=None, profiles=None,
                    provenance=None):
    """Read, regrid, and save one dropsonde dataset.

    Output dimensions: (launch_location, launch_time, altitude)
    where launch_time has size 1 (each dropsonde launched once per location).
    """
    if z_max is None:
        z_max = Z_MAX_DEFAULT

    outpath = os.path.join(OUTPUT_DIR, f"{name}.nc")
    if os.path.exists(outpath):
        print(f"\nSkipping {name} — {outpath} already exists")
        return

    if profiles is None:
        print(f"\n{'='*60}")
        print(f"Processing {name}")
        print(f"{'='*60}")

        t0 = time.time()
        profiles = reader(data_path)
        t_read = time.time() - t0
        print(f"  Read {len(profiles)} profiles in {t_read:.1f}s")

    if not profiles:
        print("  No profiles — skipping.")
        return

    edges = make_grid(Z_MIN, z_max, DZ)
    altitude = 0.5 * (edges[:-1] + edges[1:])
    n_alt = len(altitude)
    n_loc = len(profiles)

    # Pre-allocate: (launch_location, 1, altitude)
    data_arrays = {var: np.full((n_loc, 1, n_alt), np.nan) for var in DATA_VARIABLES}
    obs_time_arr = np.full((n_loc, 1, n_alt), np.datetime64("NaT"), dtype="datetime64[ns]")
    sonde_ids = []
    launch_times = np.empty((n_loc, 1), dtype="datetime64[ns]")
    launch_lats = np.full(n_loc, np.nan)
    launch_lons = np.full(n_loc, np.nan)

    t0 = time.time()
    for i, prof in enumerate(profiles):
        gridded = _regrid_profile(prof, z_max)
        for var in DATA_VARIABLES:
            if var in gridded:
                data_arrays[var][i, 0, :] = gridded[var]
        if "observation_time" in gridded:
            obs_time_arr[i, 0, :] = gridded["observation_time"]

        sonde_ids.append(prof.get("sonde_id", f"sonde_{i:06d}"))
        launch_times[i, 0] = prof.get("launch_time") or np.datetime64("NaT")
        launch_lats[i] = prof.get("launch_lat", np.nan)
        launch_lons[i] = prof.get("launch_lon", np.nan)

    t_regrid = time.time() - t0
    print(f"  Regridded in {t_regrid:.1f}s")

    dims = ("launch_location", "sounding", "altitude")
    out = xr.Dataset(
        {var: (dims, data_arrays[var]) for var in DATA_VARIABLES},
        coords={
            "altitude": altitude,
            "sonde_id": ("launch_location", sonde_ids),
            "launch_time": (("launch_location", "sounding"), launch_times),
            "launch_lat": ("launch_location", launch_lats),
            "launch_lon": ("launch_location", launch_lons),
        },
    )
    out["observation_time"] = (dims, obs_time_arr)

    _set_coord_attrs(out)
    out["sonde_id"].attrs = {"long_name": "provider sonde or profile identifier"}
    out.attrs = _global_attrs(name, z_max, n_loc, n_loc, n_alt, provenance)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out.to_netcdf(outpath)
    print(f"  Saved {outpath} ({n_loc} locations × 1 time × {n_alt} levels)")


def process_igra():
    """Process IGRA one year at a time.

    Output dimensions: (launch_location, launch_time, altitude)
    where launch_location indexes stations and launch_time indexes
    soundings at each station (padded with NaN/NaT).
    """
    data_path = os.path.join(DATA_DIR, "igra")

    for year in range(IGRA_YEAR_MIN, IGRA_YEAR_MAX + 1):
        name = f"igra_{year}"
        outpath = os.path.join(OUTPUT_DIR, f"{name}.nc")
        if os.path.exists(outpath):
            print(f"\nSkipping {name} — {outpath} already exists")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {name} (subsample={IGRA_SUBSAMPLE})")
        print(f"{'='*60}")

        t0 = time.time()
        profs = read_igra(data_path, year=year, subsample=IGRA_SUBSAMPLE)
        t_read = time.time() - t0
        print(f"  Read {len(profs)} profiles in {t_read:.1f}s")

        if not profs:
            print("  No profiles — skipping.")
            continue

        # Group profiles by station
        by_station = defaultdict(list)
        for prof in profs:
            by_station[prof["station_id"]].append(prof)

        stations = sorted(by_station.keys())
        n_stations = len(stations)
        max_times = max(len(by_station[s]) for s in stations)

        edges = make_grid(Z_MIN, Z_MAX_IGRA, DZ)
        altitude = 0.5 * (edges[:-1] + edges[1:])
        n_alt = len(altitude)

        # Pre-allocate: (n_stations, max_times, n_alt)
        data_arrays = {var: np.full((n_stations, max_times, n_alt), np.nan)
                       for var in DATA_VARIABLES}
        obs_time_arr = np.full((n_stations, max_times, n_alt),
                               np.datetime64("NaT"), dtype="datetime64[ns]")
        launch_times = np.full((n_stations, max_times), np.datetime64("NaT"),
                               dtype="datetime64[ns]")
        station_ids = []
        launch_lats = np.full(n_stations, np.nan)
        launch_lons = np.full(n_stations, np.nan)

        t0 = time.time()
        for si, station in enumerate(stations):
            station_profs = by_station[station]
            station_ids.append(station)
            launch_lats[si] = station_profs[0].get("launch_lat", np.nan)
            launch_lons[si] = station_profs[0].get("launch_lon", np.nan)

            for ti, prof in enumerate(station_profs):
                gridded = _regrid_profile(prof, Z_MAX_IGRA)
                for var in DATA_VARIABLES:
                    if var in gridded:
                        data_arrays[var][si, ti, :] = gridded[var]
                if "observation_time" in gridded:
                    obs_time_arr[si, ti, :] = gridded["observation_time"]
                launch_times[si, ti] = prof.get("launch_time") or np.datetime64("NaT")

        t_regrid = time.time() - t0
        print(f"  Regridded {len(profs)} profiles "
              f"({n_stations} stations × ≤{max_times} times) in {t_regrid:.1f}s")

        dims = ("launch_location", "sounding", "altitude")
        out = xr.Dataset(
            {var: (dims, data_arrays[var]) for var in DATA_VARIABLES},
            coords={
                "altitude": altitude,
                "station_id": ("launch_location", station_ids),
                "launch_time": (("launch_location", "sounding"), launch_times),
                "launch_lat": ("launch_location", launch_lats),
                "launch_lon": ("launch_location", launch_lons),
            },
        )
        out["observation_time"] = (dims, obs_time_arr)

        _set_coord_attrs(out, lat_long_name="latitude of station",
                         lon_long_name="longitude of station")
        out["station_id"].attrs = {"long_name": "IGRA station identifier"}
        out.attrs = _global_attrs(name, Z_MAX_IGRA, n_stations, len(profs), n_alt,
                                  DATASETS["igra"]["provenance"])

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out.to_netcdf(outpath)
        print(f"  Saved {outpath} "
              f"({n_stations} stations × {max_times} times × {n_alt} levels)")


def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

    for name in names:
        if name not in DATASETS and name != "igra":
            print(f"Unknown dataset: {name}")
            print(f"Available: {', '.join(DATASETS.keys())}")
            sys.exit(1)

    for name in names:
        if name == "igra":
            process_igra()
        else:
            cfg = DATASETS[name]
            process_dataset(name, cfg["reader"], cfg["path"],
                            z_max=cfg["z_max"], provenance=cfg.get("provenance"))

    print("\nDone.")


if __name__ == "__main__":
    main()
