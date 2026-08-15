"""
Add IBTrACS best-track position and intensity to a HAFS storm-centered
xarray Dataset.

Each frame in `ds` is identified by a `frame_link` string variable, e.g.:

    https://noaa-nws-hafs-pds.s3.amazonaws.com/hfsa/20241001/00/11l.2024100100.hfsa.storm.atm.f006.grb2

which encodes the ATCF storm number+basin ("11l"), the model
initialization time ("2024100100"), the model ("hfsa"/"hfsb"), and the
forecast hour ("f006"). The actual valid time for each frame is read
from the dataset's own `valid_time` coordinate rather than
reconstructed from the link, since that's the authoritative timestamp.

For each frame:
  1. The storm identity is parsed from `frame_link` and mapped to an
     ATCF-style storm ID (e.g. "11l" + init year 2024 -> "AL112024").
  2. That storm's IBTrACS best track (lat, lon, max wind vs. time) is
     interpolated (linearly, in time) to the frame's `valid_time`.
  3. Frames whose valid_time falls outside the best-track's start/end
     (i.e. would require extrapolation), or whose storm isn't found in
     the IBTrACS file at all, get NaN.

Result: `ds` gains two new variables, both indexed by the frame
dimension:
  - `best_track_location`: dims (frame_dim, "latlon"), degrees
  - `best_track_winds`: dims (frame_dim,), knots

Assumptions worth checking against your actual dataset (I inferred
these from the raw file bytes since xarray/netCDF4/h5py aren't
available in this sandbox and it has no network access to install
them -- I could not open the file directly to confirm at runtime):
  - The frame dimension is named "frame_number".
  - `frame_link` values look like the AWS HAFS URL pattern above.
  - `valid_time` is already decoded to datetime64 by xarray (it's
    CF-encoded as "hours since <reference>" in the file).

IBTrACS-specific notes (verified against the supplied
ibtracs_ALL_list_v04r01.csv):
  - This is the "ALL" list file, global in coverage and spanning
    1842-present, so -- unlike the old HURDAT2 (Atlantic-only,
    through-2023) file -- basin/year is no longer a source of NaNs by
    itself. Storms will only come back NaN if genuinely absent from
    the file, or if a frame's valid_time falls outside that storm's
    observed track (no extrapolation, same as before).
  - `USA_ATCF_ID` is IBTrACS' own ATCF-style ID (e.g. "AL092023") and
    matches the ID format built from `frame_link` exactly -- no
    basin-letter/format translation needed on that side. Only rows
    where a US agency (NHC/JTWC, via `hurdat_atl`/`hurdat_epa`/etc.)
    contributed are populated for this field, which is exactly the
    AL/EP/CP-basin coverage `_frame_link_to_storm_id` looks for.
  - Wind is taken from `USA_WIND` (knots, same convention as the old
    HURDAT2 `wind` column), falling back to `WMO_WIND` for obs where
    IBTrACS' own wind-speed consensus reported but USA_WIND is blank
    -- this fills in more of the track than HURDAT2's single source
    could. Lat/lon come from `LAT`/`LON`, which IBTrACS already gives
    as signed decimal degrees (no "36.4N"/"71.3W"-style parsing
    needed, unlike HURDAT2).
  - The file's second line is a units row (e.g. "degrees_north"),
    not data -- it's skipped explicitly rather than parsed.
"""

import re
import numpy as np
import pandas as pd
import xarray as xr

# ATCF basin-letter -> ATCF basin-code prefix, matching the prefix used
# in IBTrACS' USA_ATCF_ID field (e.g. "AL092023").
BASIN_LETTER_MAP = {"l": "AL", "e": "EP", "c": "CP"}

FRAME_LINK_RE = re.compile(
    r"/(\d{2})([a-z])\.(\d{10})\.(hfsa|hfsb)\.storm\.atm\.f(\d{3})\.grb2"
)


def parse_ibtracs(path: str) -> pd.DataFrame:
    """Parse an IBTrACS "list" CSV (e.g. ibtracs_ALL_list_v04r01.csv)
    into a tidy DataFrame with columns: storm_id, datetime, lat, lon,
    wind (knots, NaN where unreported).

    Only rows carrying a `USA_ATCF_ID` are kept, since that's the
    field that lines up with the AL/EP/CP-style IDs parsed from
    `frame_link`. `wind` prefers `USA_WIND` and falls back to
    `WMO_WIND` where the former is blank.
    """
    df = pd.read_csv(
        path,
        skiprows=[1],  # second line is a units row, not data
        usecols=["ISO_TIME", "LAT", "LON", "USA_ATCF_ID", "USA_WIND", "WMO_WIND"],
        na_values=[" ", ""],
        skipinitialspace=True,
        low_memory=False,
    )

    df["USA_ATCF_ID"] = df["USA_ATCF_ID"].str.strip()
    df = df[df["USA_ATCF_ID"].notna() & (df["USA_ATCF_ID"] != "")]

    out = pd.DataFrame(
        {
            "storm_id": df["USA_ATCF_ID"],
            "datetime": pd.to_datetime(df["ISO_TIME"]),
            "lat": pd.to_numeric(df["LAT"], errors="coerce"),
            "lon": pd.to_numeric(df["LON"], errors="coerce"),
            "wind": pd.to_numeric(df["USA_WIND"], errors="coerce").fillna(
                pd.to_numeric(df["WMO_WIND"], errors="coerce")
            ),
        }
    )
    return out.sort_values(["storm_id", "datetime"]).reset_index(drop=True)


def _frame_link_to_storm_id(link: str) -> str | None:
    """Extract an ATCF-style storm ID (e.g. 'AL092023') from a
    frame_link URL. Returns None if the link doesn't match the
    expected pattern or the basin letter isn't recognized.
    """
    if not isinstance(link, str):
        return None
    m = FRAME_LINK_RE.search(link)
    if m is None:
        return None
    number, basin_letter, init_dt, _model, _fhr = m.groups()
    basin = BASIN_LETTER_MAP.get(basin_letter)
    if basin is None:
        return None
    year = init_dt[:4]
    return f"{basin}{number}{year}"


def add_best_track(
    ds: xr.Dataset,
    ibtracs_path: str,
    frame_dim: str = "frame_number",
    frame_link_var: str = "frame_link",
    valid_time_var: str = "valid_time",
) -> xr.Dataset:
    """Add `best_track_location` and `best_track_winds` variables to `ds`,
    interpolated from the IBTrACS best track at `ibtracs_path` and aligned
    to each frame's storm identity (from `frame_link_var`) and timestamp
    (from `valid_time_var`).

    Returns a new Dataset (does not modify `ds` in place).
    """
    ibtracs = parse_ibtracs(ibtracs_path)

    links = ds[frame_link_var].values
    # Decode bytes if the netCDF string came back as fixed-width bytes.
    links = np.array(
        [l.decode("utf-8") if isinstance(l, bytes) else l for l in links]
    )
    storm_ids = np.array([_frame_link_to_storm_id(l) for l in links])

    valid_times = pd.to_datetime(ds[valid_time_var].values)
    valid_times_RI = valid_times + pd.Timedelta(24, unit = 'hours')

    n = len(links)
    best_lat = np.full(n, np.nan)
    best_lon = np.full(n, np.nan)
    best_wind = np.full(n, np.nan)
    best_lat_RI = np.full(n, np.nan)
    best_lon_RI = np.full(n, np.nan)
    best_wind_RI = np.full(n, np.nan)

    for storm_id in pd.unique(storm_ids):
        print(f'Storm id: {storm_id}')
        if storm_id is None:
            
            continue
        track = ibtracs.loc[ibtracs.storm_id == storm_id, ["datetime", "lat", "lon", "wind"]]
        if track.empty:
            print('Storm id not in IBTracs file')
            continue  # storm not present in this IBTrACS file (e.g. no US-agency track)

        track = track.drop_duplicates(subset="datetime").set_index("datetime").sort_index()
        RI_track = track.copy()

        frame_mask = storm_ids == storm_id
        frame_idx = np.where(frame_mask)[0]
        frame_times = pd.DatetimeIndex(valid_times[frame_idx])
        frame_times_RI = pd.DatetimeIndex(valid_times_RI[frame_idx])

        # Interpolate only *between* known best-track points (no
        # extrapolation before the first or after the last ob).
        combined_index = track.index.union(frame_times).drop_duplicates()
        RI_index = RI_track.index.union(frame_times_RI).drop_duplicates()
        interpolated = track.reindex(combined_index).interpolate(
            method="time", limit_area="inside"
        )
        interpolated_RI = RI_track.reindex(RI_index).interpolate(
                    method="time", limit_area="inside"
                )
        aligned = interpolated.loc[frame_times]
        RI_aligned = interpolated_RI.loc[frame_times_RI]
        best_lat[frame_idx] = aligned["lat"].to_numpy()
        best_lon[frame_idx] = aligned["lon"].to_numpy()
        best_wind[frame_idx] = aligned["wind"].to_numpy()

        best_lat_RI[frame_idx] = RI_aligned["lat"].to_numpy()
        best_lon_RI[frame_idx] = RI_aligned["lon"].to_numpy()
        best_wind_RI[frame_idx] = RI_aligned["wind"].to_numpy()


    location = np.stack([best_lat, best_lon], axis=-1)  # (n_frames, 2)
    location_RI = np.stack([best_lat_RI, best_lon_RI], axis=-1)  # (n_frames, 2)

    ds = ds.assign_coords(latlon=["lat", "lon"])
    ds = ds.assign_coords(latlonRI = ["lat", "lon"])
    ds["best_track_location"] = (
        (frame_dim, "latlon"),
        location,
        {
            "long_name": "IBTrACS best track storm center location, interpolated to valid_time",
            "units": "degrees",
        },
    )
    ds["best_track_winds"] = (
        (frame_dim,),
        best_wind,
        {
            "long_name": "IBTrACS best track maximum sustained wind, interpolated to valid_time",
            "units": "kt",
        },
    )
    ds["best_track_RI_location"] = (
        (frame_dim, "latlonRI"),
        location_RI,
        {
            "long_name": "IBTrACS best track storm center location in 24 hours, interpolated to valid_time",
            "units": "degrees",
        },
    )
    ds["best_track_24winds"] = (
        (frame_dim,),
        best_wind_RI,
        {
            "long_name": "IBTrACS best track maximum sustained wind, interpolated to valid_time",
            "units": "kt",
        },
    )
    return ds


if __name__ == "__main__":
    ds = xr.open_dataset("claude_ds.nc")
    ds = add_best_track(ds, "ibtracs_ALL_list_v04r01.csv")
    print(ds[["best_track_location", "best_track_winds"]])
    # ds.to_netcdf("claude_ds_with_best_track.nc")
