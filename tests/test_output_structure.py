"""
Validate the structure and metadata of all generated output NetCDF files.

Loops over every .nc file in output/ and checks that dimensions,
coordinates, data variables, and attributes match what process.py produces.
These tests operate on the generated datasets themselves — they are not
useful unless the datasets have been regenerated after code changes.
"""

import os
import glob

import numpy as np
import pytest
import xarray as xr

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

DATA_VARIABLES = ["u", "v", "p", "T", "RH", "q", "theta", "theta_e", "MSE", "DSE"]

EXPECTED_UNITS = {
    "u": "m s-1",
    "v": "m s-1",
    "p": "Pa",
    "T": "K",
    "RH": "%",
    "q": "kg kg-1",
    "theta": "K",
    "theta_e": "K",
    "MSE": "J kg-1",
    "DSE": "J kg-1",
}

REQUIRED_GLOBAL_ATTRS = [
    "title", "source", "history", "Conventions", "regridding_method",
    "grid_spacing_m", "grid_min_m", "grid_max_m",
    "n_launch_locations", "n_soundings", "n_altitude_levels",
]


def _find_output_files():
    """Return list of (name, path) for all .nc files in output/."""
    paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.nc")))
    return [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]


OUTPUT_FILES = _find_output_files()

if not OUTPUT_FILES:
    pytest.skip("No output files found in output/", allow_module_level=True)


@pytest.fixture(params=OUTPUT_FILES, ids=[name for name, _ in OUTPUT_FILES])
def dataset(request):
    name, path = request.param
    ds = xr.open_dataset(path)
    yield name, ds
    ds.close()


def _is_igra(name):
    return name.startswith("igra")


class TestDimensions:
    def test_has_three_dimensions(self, dataset):
        name, ds = dataset
        assert set(ds.dims) == {"launch_location", "sounding", "altitude"}

    def test_altitude_is_uniform(self, dataset):
        name, ds = dataset
        alt = ds["altitude"].values
        dz = np.diff(alt)
        assert np.allclose(dz, dz[0]), "altitude grid is not uniformly spaced"

    def test_altitude_positive_up(self, dataset):
        name, ds = dataset
        alt = ds["altitude"].values
        assert alt[0] < alt[-1], "altitude should increase"

    def test_dropsonde_sounding_dim_is_1(self, dataset):
        name, ds = dataset
        if _is_igra(name):
            pytest.skip("IGRA has multiple soundings per station")
        assert ds.sizes["sounding"] == 1


class TestCoordinates:
    def test_has_launch_lat(self, dataset):
        _, ds = dataset
        assert "launch_lat" in ds.coords
        assert ds["launch_lat"].dims == ("launch_location",)
        assert ds["launch_lat"].attrs.get("units") == "degrees_north"

    def test_has_launch_lon(self, dataset):
        _, ds = dataset
        assert "launch_lon" in ds.coords
        assert ds["launch_lon"].dims == ("launch_location",)
        assert ds["launch_lon"].attrs.get("units") == "degrees_east"

    def test_has_launch_time(self, dataset):
        _, ds = dataset
        assert "launch_time" in ds
        assert set(ds["launch_time"].dims) == {"launch_location", "sounding"}
        assert np.issubdtype(ds["launch_time"].dtype, np.datetime64)
        assert "long_name" in ds["launch_time"].attrs

    def test_has_observation_time(self, dataset):
        _, ds = dataset
        assert "observation_time" in ds
        assert set(ds["observation_time"].dims) == {"launch_location", "sounding", "altitude"}
        assert np.issubdtype(ds["observation_time"].dtype, np.datetime64)
        assert "long_name" in ds["observation_time"].attrs

    def test_has_location_id(self, dataset):
        name, ds = dataset
        if _is_igra(name):
            assert "station_id" in ds.coords
        else:
            assert "sonde_id" in ds.coords

    def test_altitude_attrs(self, dataset):
        _, ds = dataset
        attrs = ds["altitude"].attrs
        assert attrs.get("units") == "m"
        assert attrs.get("positive") == "up"

    def test_lat_lon_ranges(self, dataset):
        _, ds = dataset
        lats = ds["launch_lat"].values
        lons = ds["launch_lon"].values
        finite_lats = lats[np.isfinite(lats)]
        finite_lons = lons[np.isfinite(lons)]
        assert len(finite_lats) > 0, "no finite latitudes at all"
        assert len(finite_lons) > 0, "no finite longitudes at all"
        assert np.all((-90 <= finite_lats) & (finite_lats <= 90)), "latitude out of [-90, 90]"
        assert np.all((-360 <= finite_lons) & (finite_lons <= 360)), "longitude out of [-360, 360]"

    def test_profile_duration_under_6h(self, dataset):
        """All profiles should span less than 6 hours from launch to last observation."""
        _, ds = dataset
        max_duration = np.timedelta64(6, "h")
        launch = ds["launch_time"].values        # (location, sounding)
        obs = ds["observation_time"].values       # (location, sounding, altitude)
        n_loc, n_snd = launch.shape
        for i in range(n_loc):
            for j in range(n_snd):
                lt = launch[i, j]
                if np.isnat(lt):
                    continue
                obs_ij = obs[i, j, :]
                valid = obs_ij[~np.isnat(obs_ij)]
                if len(valid) == 0:
                    continue
                all_times = np.concatenate([[lt], valid])
                span = all_times.max() - all_times.min()
                assert span < max_duration, (
                    f"profile ({i}, {j}) spans {span} "
                    f"(launch {lt}, obs {all_times.min()} to {all_times.max()})"
                )


class TestDataVariables:
    def test_all_data_variables_present(self, dataset):
        _, ds = dataset
        for var in DATA_VARIABLES:
            assert var in ds, f"missing data variable {var}"

    def test_data_variable_shape(self, dataset):
        _, ds = dataset
        expected_shape = (
            ds.sizes["launch_location"],
            ds.sizes["sounding"],
            ds.sizes["altitude"],
        )
        for var in DATA_VARIABLES:
            assert ds[var].shape == expected_shape, f"{var} shape mismatch"

    def test_data_variable_units(self, dataset):
        _, ds = dataset
        for var, expected_unit in EXPECTED_UNITS.items():
            assert ds[var].attrs.get("units") == expected_unit, \
                f"{var} units: expected {expected_unit!r}, got {ds[var].attrs.get('units')!r}"

    def test_data_variables_have_long_name(self, dataset):
        _, ds = dataset
        for var in DATA_VARIABLES:
            assert "long_name" in ds[var].attrs, f"{var} missing long_name"


class TestPhysicalPlausibility:
    """Check that values fall within physically realistic ranges."""

    def test_temperature_range(self, dataset):
        _, ds = dataset
        T = ds["T"].values
        valid = T[np.isfinite(T)]
        if len(valid) == 0:
            pytest.skip("no valid temperature data")
        assert np.all((180 <= valid) & (valid <= 330)), \
            f"temperature out of [180, 330] K: min={valid.min():.1f}, max={valid.max():.1f}"

    def test_pressure_decreases_with_altitude(self, dataset):
        """Mean pressure in the lower half should exceed mean pressure in the upper half."""
        _, ds = dataset
        p = ds["p"].values  # (location, sounding, altitude)
        n_loc, n_snd, n_alt = p.shape
        mid = n_alt // 2
        for i in range(n_loc):
            for j in range(n_snd):
                lower = p[i, j, :mid]
                upper = p[i, j, mid:]
                lower_valid = lower[np.isfinite(lower)]
                upper_valid = upper[np.isfinite(upper)]
                if len(lower_valid) == 0 or len(upper_valid) == 0:
                    continue
                assert lower_valid.mean() > upper_valid.mean(), (
                    f"profile ({i}, {j}): mean pressure in lower half "
                    f"({lower_valid.mean():.0f} Pa) <= upper half "
                    f"({upper_valid.mean():.0f} Pa)"
                )

    def test_rh_range(self, dataset):
        _, ds = dataset
        rh = ds["RH"].values
        valid = rh[np.isfinite(rh)]
        if len(valid) == 0:
            pytest.skip("no valid RH data")
        assert np.all((0 <= valid) & (valid <= 100)), \
            f"RH out of [0, 100] %: min={valid.min():.1f}, max={valid.max():.1f}"

    def test_specific_humidity_range(self, dataset):
        _, ds = dataset
        q = ds["q"].values
        valid = q[np.isfinite(q)]
        if len(valid) == 0:
            pytest.skip("no valid q data")
        assert np.all((0 <= valid) & (valid <= 0.05)), \
            f"q out of [0, 0.05] kg/kg: min={valid.min():.6f}, max={valid.max():.6f}"

    def test_theta_e_exceeds_theta(self, dataset):
        _, ds = dataset
        theta_e = ds["theta_e"].values
        theta = ds["theta"].values
        both_valid = np.isfinite(theta_e) & np.isfinite(theta)
        if not np.any(both_valid):
            pytest.skip("no co-valid theta_e and theta data")
        assert np.all(theta_e[both_valid] >= theta[both_valid]), \
            "theta_e < theta found; equivalent potential temperature must exceed theta"

    def test_mse_exceeds_dse(self, dataset):
        _, ds = dataset
        mse = ds["MSE"].values
        dse = ds["DSE"].values
        both_valid = np.isfinite(mse) & np.isfinite(dse)
        if not np.any(both_valid):
            pytest.skip("no co-valid MSE and DSE data")
        assert np.all(mse[both_valid] >= dse[both_valid]), \
            "MSE < DSE found; MSE = DSE + Lv*q so MSE >= DSE"

    def test_wind_range(self, dataset):
        _, ds = dataset
        for comp in ("u", "v"):
            vals = ds[comp].values
            valid = vals[np.isfinite(vals)]
            if len(valid) == 0:
                continue
            assert np.all(np.abs(valid) < 150), \
                f"|{comp}| >= 150 m/s found: max={np.abs(valid).max():.1f}"


class TestDataCoverage:
    """Check that profiles contain meaningful data."""

    def test_each_profile_has_some_data(self, dataset):
        """Every profile should have at least one non-NaN value across all variables."""
        _, ds = dataset
        n_loc, n_snd, _ = ds["T"].shape
        for i in range(n_loc):
            for j in range(n_snd):
                any_valid = False
                for var in DATA_VARIABLES:
                    col = ds[var].values[i, j, :]
                    if np.any(np.isfinite(col)):
                        any_valid = True
                        break
                assert any_valid, (
                    f"profile ({i}, {j}) is entirely NaN across all variables"
                )

    def test_observation_time_overall_trend(self, dataset):
        """observation_time should trend consistently along altitude (first vs last)."""
        _, ds = dataset
        obs = ds["observation_time"].values  # (location, sounding, altitude)
        n_loc, n_snd, _ = obs.shape
        for i in range(n_loc):
            for j in range(n_snd):
                col = obs[i, j, :]
                valid_mask = ~np.isnat(col)
                if valid_mask.sum() < 5:
                    continue
                valid_times = col[valid_mask]
                # Overall direction should be consistent: first and last
                # should differ by more than any local jitter
                span = valid_times[-1] - valid_times[0]
                assert span != np.timedelta64(0), (
                    f"profile ({i}, {j}): all observation_times are identical"
                )


class TestCoordinateConsistency:
    """Cross-check coordinates for internal consistency."""

    def test_launch_time_near_first_obs(self, dataset):
        """launch_time should be within 90 minutes of the earliest observation_time.

        Most datasets have <10 min gaps; DYNAMO uses a time_offset convention
        where the reference time is 1 hour after the first measurement.
        """
        _, ds = dataset
        launch = ds["launch_time"].values       # (location, sounding)
        obs = ds["observation_time"].values      # (location, sounding, altitude)
        tolerance = np.timedelta64(90, "m")
        n_loc, n_snd = launch.shape
        for i in range(n_loc):
            for j in range(n_snd):
                lt = launch[i, j]
                if np.isnat(lt):
                    continue
                obs_ij = obs[i, j, :]
                valid = obs_ij[~np.isnat(obs_ij)]
                if len(valid) == 0:
                    continue
                first_obs = valid.min()
                gap = abs(lt - first_obs)
                assert gap <= tolerance, (
                    f"profile ({i}, {j}): launch_time {lt} is {gap} from "
                    f"first observation {first_obs} (tolerance: {tolerance})"
                )

    def test_unique_sonde_ids(self, dataset):
        name, ds = dataset
        if _is_igra(name):
            ids = ds["station_id"].values
        else:
            ids = ds["sonde_id"].values
        id_list = list(ids)
        assert len(id_list) == len(set(id_list)), "duplicate sonde/station IDs found"



class TestGlobalAttributes:
    def test_required_global_attrs(self, dataset):
        _, ds = dataset
        for attr in REQUIRED_GLOBAL_ATTRS:
            assert attr in ds.attrs, f"missing global attribute {attr!r}"

    def test_conventions(self, dataset):
        _, ds = dataset
        assert ds.attrs["Conventions"] == "CF-1.8"

    def test_regridding_method(self, dataset):
        _, ds = dataset
        assert "bin averaging" in ds.attrs["regridding_method"]

    def test_provenance_attrs(self, dataset):
        _, ds = dataset
        assert "source_campaign" in ds.attrs, "missing provenance: source_campaign"
        assert "source_citation" in ds.attrs, "missing provenance: source_citation"

    def test_grid_params_consistent_with_altitude(self, dataset):
        _, ds = dataset
        alt = ds["altitude"].values
        dz = ds.attrs["grid_spacing_m"]
        assert np.isclose(np.diff(alt).mean(), dz, rtol=0.01)
        assert alt[0] >= ds.attrs["grid_min_m"]
        assert alt[-1] <= ds.attrs["grid_max_m"]
