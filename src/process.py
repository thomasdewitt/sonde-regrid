"""
Process all datasets: read → regrid → diagnose → write NetCDF.

Produces one NetCDF per dropsonde dataset and one per IGRA station.
Output dimensions are (sounding_id, altitude).

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

IGRA_YEAR_MIN = 2000
IGRA_YEAR_MAX = 2025
IGRA_SUBSAMPLE = 1        # keep every Nth day (1 = all)

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
            "period": "2000-2025",
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
    out["launch_time"].attrs = {
        "long_name": "date and time of sounding launch",
        "comment": "For IGRA: parsed from RELTIME field (actual release time in HHMM). "
                   "When RELTIME is missing (9999) or invalid, falls back to the nominal "
                   "synoptic hour. If RELTIME places the launch >12h after the nominal "
                   "time, the date is shifted back one day (pre-midnight launches for 00Z).",
    }
    out["observation_time"].attrs = {
        "long_name": "observation time at each altitude level",
        "comment": "Bin-averaged within each grid cell. Computed as launch_time + "
                   "elapsed time from the sounding data.",
    }
    if "nominal_time" in out:
        out["nominal_time"].attrs = {
            "long_name": "nominal synoptic time (0, 6, 12, or 18 UTC)",
            "comment": "The synoptic hour from the IGRA header (HR field). "
                       "Differs from launch_time, which is the actual release time.",
        }
    for var, attrs in VARIABLE_ATTRS.items():
        if var in out:
            out[var].attrs = attrs


def _global_attrs(name, z_max, n_soundings, n_alt, provenance=None):
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
        "n_soundings": n_soundings,
        "n_altitude_levels": n_alt,
        "variables_directly_gridded": "u, v, p, T, RH",
        "variables_diagnosed_post_gridding": "q, theta, theta_e, MSE, DSE",
        "missing_value_note": "NaN indicates grid cells with no valid observations",
        "profile_filtering": "Profiles with no finite altitude values are excluded "
                             "prior to regridding (cannot be placed on the grid).",
    }
    if provenance:
        for key, val in provenance.items():
            attrs[f"source_{key}"] = val
    return attrs


def process_dataset(name, reader, data_path, z_max=None, profiles=None,
                    provenance=None):
    """Read, regrid, and save one dataset.

    Output dimensions: (sounding_id, altitude).
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

    # Drop profiles that can't be regridded (no valid altitude values)
    def _has_data(p):
        alt = p.get("altitude")
        if alt is None or not np.any(np.isfinite(alt)):
            return False
        return True

    n_before = len(profiles)
    profiles = [p for p in profiles if _has_data(p)]
    if len(profiles) < n_before:
        print(f"  Dropped {n_before - len(profiles)} empty profiles")
    if not profiles:
        print("  No profiles with valid data — skipping.")
        return

    edges = make_grid(Z_MIN, z_max, DZ)
    altitude = 0.5 * (edges[:-1] + edges[1:])
    n_alt = len(altitude)
    n_prof = len(profiles)

    data_arrays = {var: np.full((n_prof, n_alt), np.nan) for var in DATA_VARIABLES}
    obs_time_arr = np.full((n_prof, n_alt), np.datetime64("NaT"), dtype="datetime64[ns]")
    sonde_ids = []
    launch_times = np.empty(n_prof, dtype="datetime64[ns]")
    launch_lats = np.full(n_prof, np.nan)
    launch_lons = np.full(n_prof, np.nan)
    has_nominal = any("nominal_time" in p for p in profiles)
    if has_nominal:
        nominal_times = np.empty(n_prof, dtype="datetime64[ns]")

    t0 = time.time()
    for i, prof in enumerate(profiles):
        gridded = _regrid_profile(prof, z_max)
        for var in DATA_VARIABLES:
            if var in gridded:
                data_arrays[var][i, :] = gridded[var]
        if "observation_time" in gridded:
            obs_time_arr[i, :] = gridded["observation_time"]

        sonde_ids.append(prof.get("sonde_id", f"sonde_{i:06d}"))
        launch_times[i] = prof.get("launch_time") or np.datetime64("NaT")
        launch_lats[i] = prof.get("launch_lat", np.nan)
        launch_lons[i] = prof.get("launch_lon", np.nan)
        if has_nominal:
            nominal_times[i] = prof.get("nominal_time") or np.datetime64("NaT")

    t_regrid = time.time() - t0
    print(f"  Regridded {n_prof} profiles in {t_regrid:.1f}s")

    dims = ("sounding_id", "altitude")
    coords = {
        "altitude": altitude,
        "sonde_id": ("sounding_id", sonde_ids),
        "launch_time": ("sounding_id", launch_times),
        "launch_lat": ("sounding_id", launch_lats),
        "launch_lon": ("sounding_id", launch_lons),
    }
    if has_nominal:
        coords["nominal_time"] = ("sounding_id", nominal_times)
    out = xr.Dataset(
        {var: (dims, data_arrays[var]) for var in DATA_VARIABLES},
        coords=coords,
    )
    out["observation_time"] = (dims, obs_time_arr)

    _set_coord_attrs(out)
    out["sonde_id"].attrs = {"long_name": "provider sonde or profile identifier"}
    out.attrs = _global_attrs(name, z_max, n_prof, n_alt, provenance)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    out.to_netcdf(outpath)
    print(f"  Saved {outpath} ({n_prof} soundings × {n_alt} levels)")


def _read_igra_metadata(metadata_path):
    """Parse igra2-metadata.txt into a dict keyed by station ID.

    Returns dict[station_id] with keys: station_name, wmo_id, latitude,
    longitude, elevation, equipment_history (full multi-line string).
    """
    stations = {}
    with open(metadata_path) as f:
        for line in f:
            if len(line.strip()) < 50:
                continue
            station_id = line[0:11].strip()
            wmo_id = line[12:17].strip()
            station_name = line[18:48].strip()
            try:
                lat = float(line[49:60])
                lon = float(line[60:72])
                elev = float(line[72:82])
            except ValueError:
                lat = lon = elev = None

            if station_id not in stations:
                stations[station_id] = {
                    "station_name": station_name,
                    "wmo_id": wmo_id,
                    "latitude": lat,
                    "longitude": lon,
                    "elevation": elev,
                    "equipment_history": line.rstrip(),
                }
            else:
                # Update to most recent entry; append history
                stations[station_id]["station_name"] = station_name
                if lat is not None:
                    stations[station_id]["latitude"] = lat
                    stations[station_id]["longitude"] = lon
                    stations[station_id]["elevation"] = elev
                stations[station_id]["equipment_history"] += "\n" + line.rstrip()
    return stations


def _igra_provenance(metadata, station_id):
    """Build provenance dict for one IGRA station."""
    provenance = {**DATASETS["igra"]["provenance"]}
    station_meta = metadata.get(station_id, {})
    if station_meta:
        provenance["station_name"] = station_meta["station_name"]
        provenance["wmo_id"] = station_meta["wmo_id"]
        provenance["station_elevation_m"] = (
            str(station_meta["elevation"]) if station_meta["elevation"] is not None
            else "unknown")
        provenance["equipment_history"] = station_meta["equipment_history"]
        provenance["metadata_note"] = (
            "Station name, location, and elevation are from the most recent entry "
            "in igra2-metadata.txt. Equipment history includes all catalog entries.")
    return provenance


def process_igra(stations=None):
    """Process IGRA: one file per station, all years combined.

    Reads and writes one station at a time to avoid loading the entire
    archive into memory.

    Parameters
    ----------
    stations : list of str, optional
        Station IDs to process. If None, process all stations.

    Output dimensions: (sounding_id, altitude).
    """
    data_path = os.path.join(DATA_DIR, "igra")
    metadata = _read_igra_metadata(os.path.join(data_path, "igra2-metadata.txt"))

    print(f"\n{'='*60}")
    print(f"Processing IGRA ({IGRA_YEAR_MIN}-{IGRA_YEAR_MAX})")
    print(f"{'='*60}")

    # Discover station zip files
    pattern = os.path.join(data_path, "*-data.txt.zip")
    import glob as _glob
    zips = sorted(_glob.glob(pattern))
    station_ids = [os.path.basename(z).split("-data")[0] for z in zips]

    if stations is not None:
        station_ids = [s for s in station_ids if s in set(stations)]

    # Skip stations whose output already exists
    todo = []
    for sid in station_ids:
        outpath = os.path.join(OUTPUT_DIR, f"igra/{sid}.nc")
        if os.path.exists(outpath):
            print(f"  Skipping igra/{sid} — already exists")
        else:
            todo.append(sid)

    print(f"  {len(todo)} stations to process")

    for sid in todo:
        profs = read_igra(data_path, year_min=IGRA_YEAR_MIN, year_max=IGRA_YEAR_MAX,
                          subsample=IGRA_SUBSAMPLE, stations=[sid])
        if not profs:
            continue
        process_dataset(
            f"igra/{sid}", reader=None, data_path=None,
            z_max=Z_MAX_IGRA,
            profiles=profs,
            provenance=_igra_provenance(metadata, sid),
        )


def main():
    """Process datasets specified on the command line.

    Usage:
        python process.py                          # all datasets
        python process.py joanne otrec             # specific dropsonde datasets
        python process.py igra                     # all IGRA stations
        python process.py igra:USM00072451,USM00072764  # specific IGRA stations
    """
    args = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

    # Parse igra:STATION1,STATION2 syntax (multiple igra: args are merged)
    names = []
    igra_stations = None
    for arg in args:
        if arg.startswith("igra:"):
            if "igra" not in names:
                names.append("igra")
            these = arg.split(":", 1)[1].split(",")
            igra_stations = (igra_stations or []) + these
        else:
            names.append(arg)

    for name in names:
        if name not in DATASETS and name != "igra":
            print(f"Unknown dataset: {name}")
            print(f"Available: {', '.join(DATASETS.keys())}")
            sys.exit(1)

    for name in names:
        if name == "igra":
            process_igra(stations=igra_stations)
        else:
            cfg = DATASETS[name]
            process_dataset(name, cfg["reader"], cfg["path"],
                            z_max=cfg["z_max"], provenance=cfg.get("provenance"))

    print("\nDone.")


if __name__ == "__main__":
    main()
