import re
import io
import ssl
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd


# --------------------------------------------------
# settings
# --------------------------------------------------
years = range(2020, 2026)

base_dir = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
index_url = base_dir

# This avoids the SSL certificate error on your local Python.
# It is okay here because we are just downloading public NOAA csv files.
ssl_context = ssl._create_unverified_context()


# --------------------------------------------------
# helpers
# --------------------------------------------------
def download_url(url, timeout=60):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout, context=ssl_context) as resp:
        return resp.read()


def find_detail_file_for_year(year, html_text):
    """
    Find the latest details file for a given year from the NOAA directory index.
    Example filename:
    StormEvents_details-ftp_v1.0_d2024_c20260323.csv.gz
    """
    pattern = rf"StormEvents_details-ftp_v1\.0_d{year}_c\d{{8}}\.csv\.gz"
    matches = re.findall(pattern, html_text)

    if not matches:
        return None

    return max(matches)


def parse_damage(x):
    """
    Convert NOAA damage strings like 10.00K, 2.50M, 1.00B to numeric dollars.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().upper()

    if s == "" or s == "0" or s == "0.00K":
        return 0.0

    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([KMB]?)", s)

    if m is None:
        return np.nan

    val = float(m.group(1))
    suffix = m.group(2)

    mult = {
        "": 1.0,
        "K": 1e3,
        "M": 1e6,
        "B": 1e9,
    }[suffix]

    return val * mult


def ef_to_int(x):
    """
    Convert EF-scale strings like EF0, EF1, ..., EF5 to integers.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().upper()
    m = re.fullmatch(r"EF([0-5])", s)

    if m:
        return int(m.group(1))

    return np.nan


# --------------------------------------------------
# get NOAA index page
# --------------------------------------------------
html = download_url(index_url).decode("utf-8", errors="ignore")


# --------------------------------------------------
# download and stack tornado rows
# --------------------------------------------------
frames = []

for year in years:
    fname = find_detail_file_for_year(year, html)

    if fname is None:
        print(f"[skip] no file found for {year}")
        continue

    url = base_dir + fname
    print(f"[read] {url}")

    data = download_url(url)

    df = pd.read_csv(
        io.BytesIO(data),
        compression="gzip",
        low_memory=False
    )

    df.columns = [c.lower() for c in df.columns]

    if "event_type" not in df.columns:
        print(f"[skip] event_type not found for {year}")
        continue

    tor = df[
        df["event_type"].astype(str).str.strip().str.lower() == "tornado"
    ].copy()

    tor["source_file"] = fname
    tor["data_year"] = year

    frames.append(tor)


if not frames:
    raise ValueError("No tornado rows were loaded.")


tornado = pd.concat(frames, ignore_index=True)


# --------------------------------------------------
# keep useful columns if present
# --------------------------------------------------
keep_cols = [
    "event_id",
    "episode_id",
    "year",
    "month_name",
    "state",
    "state_fips",
    "cz_name",
    "cz_type",
    "cz_fips",
    "wfo",
    "begin_date_time",
    "end_date_time",
    "cz_timezone",
    "begin_lat",
    "begin_lon",
    "end_lat",
    "end_lon",
    "injuries_direct",
    "injuries_indirect",
    "deaths_direct",
    "deaths_indirect",
    "damage_property",
    "damage_crops",
    "tor_f_scale",
    "tor_length",
    "tor_width",
    "source",
    "event_narrative",
    "episode_narrative",
    "source_file",
    "data_year",
]

keep_cols = [c for c in keep_cols if c in tornado.columns]
tornado = tornado[keep_cols].copy()


# --------------------------------------------------
# parse datetimes
# --------------------------------------------------
for col in ["begin_date_time", "end_date_time"]:
    if col in tornado.columns:
        tornado[col] = pd.to_datetime(tornado[col], errors="coerce")


# --------------------------------------------------
# numeric cleanup
# --------------------------------------------------
numeric_cols = [
    "begin_lat",
    "begin_lon",
    "end_lat",
    "end_lon",
    "injuries_direct",
    "injuries_indirect",
    "deaths_direct",
    "deaths_indirect",
    "tor_length",
    "tor_width",
]

for col in numeric_cols:
    if col in tornado.columns:
        tornado[col] = pd.to_numeric(tornado[col], errors="coerce")


if "damage_property" in tornado.columns:
    tornado["damage_property_num"] = tornado["damage_property"].map(parse_damage)


if "damage_crops" in tornado.columns:
    tornado["damage_crops_num"] = tornado["damage_crops"].map(parse_damage)


if "tor_f_scale" in tornado.columns:
    tornado["tor_f_scale_num"] = tornado["tor_f_scale"].map(ef_to_int)


# --------------------------------------------------
# build a simple PPP table
# one point per tornado row, using the begin point
# --------------------------------------------------
required = [
    c for c in ["begin_date_time", "begin_lon", "begin_lat"]
    if c in tornado.columns
]

ppp = tornado.dropna(subset=required).copy()

ppp = ppp.rename(
    columns={
        "begin_date_time": "time",
        "begin_lon": "start_lon",
        "begin_lat": "start_lat",
        "end_lon": "end_lon",
        "end_lat": "end_lat",
    }
)

ppp_cols = [
    "event_id",
    "time",
    "start_lon",
    "start_lat",
    "end_lon",
    "end_lat",
    "state",
    "cz_name",
    "cz_timezone",
    "tor_f_scale",
    "tor_f_scale_num",
    "tor_length",
    "tor_width",
    "deaths_direct",
    "injuries_direct",
    "damage_property_num",
    "source_file",
    "data_year",
]

ppp_cols = [c for c in ppp_cols if c in ppp.columns]
ppp = ppp[ppp_cols].copy()


# --------------------------------------------------
# sort and save
# --------------------------------------------------
if "time" in ppp.columns:
    ppp = ppp.sort_values("time").reset_index(drop=True)


tornado.to_csv("tornado_details_all.csv", index=False)
ppp.to_csv("tornado_ppp.csv", index=False)


print("\nDone.")
print("Full tornado table shape:", tornado.shape)
print("PPP table shape:", ppp.shape)

print("\nSaved:")
print("  tornado_details_all.csv")
print("  tornado_ppp.csv")

print("\nPPP preview:")
print(ppp.head())

#########
#########
#########
ppp.to_csv("ppp.csv", index=False)
