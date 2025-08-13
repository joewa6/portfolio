# %% [markdown]
# Lines, Curves & Equilibrium — BTC smoothing + regime score
# ----------------------------------------------------------
# This script/notebook:
# 1) pulls 5m BTCUSDT candles from Binance,
# 2) applies rolling and Savitzky–Golay smoothing,
# 3) computes derivatives (slope/curvature),
# 4) computes a simple "equilibrium score" via stationarity tests,
# 5) exports clean figures for the blog.

# %% Imports & config
import os, time, math, json, pathlib, datetime as dt
from typing import Tuple
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import adfuller

# Paths
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "crypto"
FIGS = ROOT / "assets" / "img" / "blog" / "lines-curves-equilibrium"
DATA.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# Plot defaults (matplotlib: one plot per figure; no custom colors)
plt.rcParams.update({
    "figure.figsize": (10, 4.2),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 150,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

# %% Helper: fetch recent 5m klines from Binance
BINANCE = "https://api.binance.com/api/v3/klines"

def fetch_binance_klines(symbol="BTCUSDT", interval="5m", limit=1000) -> pd.DataFrame:
    """Fetch last `limit` klines for symbol/interval."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(BINANCE, params=params, timeout=20)
    r.raise_for_status()
    rows = r.json()
    # Columns per Binance docs
    cols = ["open_time","open","high","low","close","volume","close_time",
            "qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert("Europe/Rome")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.tz_convert("Europe/Rome")
    return df[["open_time","open","high","low","close","volume","close_time"]]

# %% Pull or load cache
cache = DATA / "btc_5m_latest.csv"
try:
    df = fetch_binance_klines(limit=1000)
    df.to_csv(cache, index=False)
except Exception as e:
    print("Fetch failed, falling back to cache:", e)
    assert cache.exists(), "No cache found; run again with internet."
    df = pd.read_csv(cache, parse_dates=["open_time","close_time"])

df = df.sort_values("open_time").reset_index(drop=True)
df["mid"] = (df["high"] + df["low"]) / 2.0

# %% Smoothing
# windows in number of 5m bars (e.g., 12 = 1 hour)
roll_w = 12
sg_w   = 21   # must be odd
sg_poly = 3

s_roll = df["mid"].rolling(roll_w, min_periods=roll_w//2).mean()
s_sg   = pd.Series(savgol_filter(df["mid"].to_numpy(), window_length=sg_w, polyorder=sg_poly), index=df.index)

# Derivatives from Savitzky–Golay (finite-diff fallback if needed)
# spacing: 5 minutes in seconds (for slope units per second)
dt_sec = 5 * 60
s_sg_slope = pd.Series(savgol_filter(df["mid"].to_numpy(), sg_w, sg_poly, deriv=1, delta=dt_sec), index=df.index)
s_sg_curv  = pd.Series(savgol_filter(df["mid"].to_numpy(), sg_w, sg_poly, deriv=2, delta=dt_sec), index=df.index)

# %% Plot: price + smoothers
fig1 = plt.figure()
plt.plot(df["open_time"], df["mid"], label="Mid price")
plt.plot(df["open_time"], s_roll, label=f"Rolling mean ({roll_w}×5m)")
plt.plot(df["open_time"], s_sg, label=f"Savitzky–Golay (w={sg_w}, p={sg_poly})")
plt.title("BTC/USDT 5m — raw & smoothed")
plt.xlabel("Time (Europe/Rome)")
plt.ylabel("Price (USDT)")
plt.legend()
fig1.savefig(FIGS / "price_smoothing.png", bbox_inches="tight")

# %% Plot: derivatives
fig2 = plt.figure()
plt.plot(df["open_time"], s_sg_slope, label="Slope (1st deriv)")
plt.title("Smoothed slope (d price / dt)")
plt.xlabel("Time"); plt.ylabel("USDT per second")
fig2.savefig(FIGS / "slope.png", bbox_inches="tight")

fig3 = plt.figure()
plt.plot(df["open_time"], s_sg_curv, label="Curvature (2nd deriv)")
plt.title("Smoothed curvature (2nd derivative)")
plt.xlabel("Time"); plt.ylabel("1/s^2 scaled")
fig3.savefig(FIGS / "curvature.png", bbox_inches="tight")

# %% Equilibrium score (stationarity proxy)
# Idea: sliding windows; if distributions don't change much and ADF says stationary,
# mark as equilibrium. Score in [0,1]: higher = more equilibrium-like.

win = 36   # 3 hours of 5m bars
step = 3   # slide by 15 minutes
ks_vals = []
adf_p   = []
centers = []

prices = df["mid"].to_numpy()
for i in range(0, len(prices) - 2*win, step):
    w1 = prices[i:i+win]
    w2 = prices[i+win:i+2*win]
    # KS distance between adjacent windows
    ks = ks_2samp(w1, w2).statistic
    ks_vals.append(ks)
    # ADF p-value on w2 (lower p ⇒ reject unit root ⇒ more stationary)
    try:
        pval = adfuller(w2, autolag="AIC")[1]
    except Exception:
        pval = 1.0
    adf_p.append(pval)
    centers.append(df["open_time"].iloc[i+win])

ks_vals = np.array(ks_vals)
adf_p   = np.array(adf_p)

# Normalize KS to [0,1] (already 0..1 for two-sample), convert p to "stationarity score"
stationarity = 1.0 - np.clip(adf_p, 0, 1)
# Equilibrium score: low KS + high stationarity
eq_score = (1 - ks_vals) * 0.6 + stationarity * 0.4

eq = pd.DataFrame({"time": centers, "ks": ks_vals, "adf_stationarity": stationarity, "eq_score": eq_score})

# %% Plot equilibrium score
fig4 = plt.figure()
plt.plot(eq["time"], eq["eq_score"])
plt.title("Equilibrium score (1=more stationary/unchanged)")
plt.xlabel("Time")
plt.ylabel("Score")
fig4.savefig(FIGS / "equilibrium_score.png", bbox_inches="tight")

# %% Export quick CSV (optional)
eq.to_csv(DATA / "btc_eq_score_5m.csv", index=False)

# %% Notebook → HTML export (optional one-liner if jupyter is available)
# You can run this once you open it as a notebook, or from cli:
# jupyter nbconvert --to html notebooks/btc_smoothing.ipynb
# Or keep this file as script; VS Code can "Export as Jupyter Notebook" from the command palette.

# %% Done
print("Saved figures to:", FIGS)
