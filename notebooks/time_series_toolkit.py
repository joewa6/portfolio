# %% [markdown]
# Time-Series Toolkit (BTC demo) -> 3 blog posts + project landing figures
# - Post 1: Smoothing + peaks/troughs (Savitzky–Golay, derivatives)
# - Post 2: Statistical inefficiency (autocorr, tau_int), Neff vs N
# - Post 3: Committor-like regime probability (logistic regression on features)
# All figures saved under assets/img/blog/... and assets/img/projects/...

# %% Imports & config
import os, pathlib, math
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

plt.rcParams.update({
    "figure.figsize": (10, 4.2),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 150,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "crypto"
FIG_P1 = ROOT / "assets" / "img" / "blog" / "series-1-peaks"
FIG_P2 = ROOT / "assets" / "img" / "blog" / "series-2-inefficiency"
FIG_P3 = ROOT / "assets" / "img" / "blog" / "series-3-committor"
FIG_PROJ = ROOT / "assets" / "img" / "projects" / "time-series-toolkit"
for d in [DATA, FIG_P1, FIG_P2, FIG_P3, FIG_PROJ]:
    d.mkdir(parents=True, exist_ok=True)

# %% Data fetch (Binance 5m candles)
BINANCE = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol="BTCUSDT", interval="5m", limit=1000) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(BINANCE, params=params, timeout=20)
    r.raise_for_status()
    rows = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time",
            "qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert("Europe/Rome")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.tz_convert("Europe/Rome")
    return df[["open_time","open","high","low","close","volume","close_time"]]

cache = DATA / "btc_5m_latest.csv"
try:
    df = fetch_klines(limit=1000)
    df.to_csv(cache, index=False)
except Exception as e:
    print("Fetch failed, using cache:", e)
    assert cache.exists(), "No cache available. Re-run with internet."
    df = pd.read_csv(cache, parse_dates=["open_time","close_time"])
df = df.sort_values("open_time").reset_index(drop=True)
df["mid"] = (df["high"] + df["low"]) / 2.0

# %% ---------- Post 1: Smoothing + Peaks/Troughs ----------
# SG parameters (edit these to taste)
sg_w, sg_poly = 21, 3       # window must be odd
roll_w = 12                  # 12 bars = 1 hour for 5m candles
dt_sec = 5*60

mid = df["mid"].to_numpy()
t   = df["open_time"]

# Smoothers
sg = savgol_filter(mid, window_length=sg_w, polyorder=sg_poly)
roll = pd.Series(mid).rolling(roll_w, min_periods=roll_w//2).mean().to_numpy()

# Derivatives from SG
sg_slope = savgol_filter(mid, sg_w, sg_poly, deriv=1, delta=dt_sec)
sg_curv  = savgol_filter(mid, sg_w, sg_poly, deriv=2, delta=dt_sec)

# Peaks/troughs:
# - peaks on smoothed series
peaks, _   = find_peaks(sg, distance=6)
# - troughs by finding peaks on negative signal
troughs, _ = find_peaks(-sg, distance=6)

# Plots for Post 1
plt.figure(); plt.plot(t, mid, label="Mid")
plt.plot(t, roll, label=f"Rolling mean ({roll_w}×5m)")
plt.plot(t, sg, label=f"SG (w={sg_w}, p={sg_poly})")
plt.title("BTC 5m — raw & smoothed"); plt.xlabel("Time"); plt.ylabel("USDT"); plt.legend()
plt.savefig(FIG_P1 / "p1_raw_vs_smoothed.png", bbox_inches="tight")

plt.figure(); plt.plot(t, sg_slope); plt.title("Smoothed slope (1st deriv)"); plt.xlabel("Time"); plt.ylabel("USDT/s")
plt.savefig(FIG_P1 / "p1_slope.png", bbox_inches="tight")

plt.figure(); plt.plot(t, sg_curv); plt.title("Smoothed curvature (2nd deriv)"); plt.xlabel("Time"); plt.ylabel("1/s^2 (scaled)")
plt.savefig(FIG_P1 / "p1_curvature.png", bbox_inches="tight")

plt.figure()
plt.plot(t, sg, label="SG")
plt.scatter(t.iloc[peaks], sg[peaks], s=30, label="Peaks", marker="^")
plt.scatter(t.iloc[troughs], sg[troughs], s=30, label="Troughs", marker="v")
plt.title("Peaks & troughs on smoothed series"); plt.xlabel("Time"); plt.ylabel("USDT"); plt.legend()
plt.savefig(FIG_P1 / "p1_peaks_troughs.png", bbox_inches="tight")

# Nice hero figure for the project landing
plt.figure()
plt.plot(t, mid, alpha=0.35, label="Mid")
plt.plot(t, sg, label="SG")
plt.scatter(t.iloc[peaks], sg[peaks], s=18, marker="^")
plt.scatter(t.iloc[troughs], sg[troughs], s=18, marker="v")
plt.title("Signal processing overview"); plt.xlabel("Time"); plt.ylabel("USDT")
plt.savefig(FIG_PROJ / "hero_signal_processing.png", bbox_inches="tight")

# %% ---------- Post 2: Statistical inefficiency & Neff ----------
# Autocorrelation + integrated time (tau_int); Neff ~ N / (2*tau_int+1)
def autocorr(x):
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(np.concatenate([x, np.zeros_like(x)]))
    acf = np.fft.irfft(f * np.conjugate(f))[:n]
    acf /= acf[0]
    return acf

def tau_int(acf, cutoff="auto"):
    # Truncate when acf becomes negative or below small threshold
    if cutoff == "auto":
        # windowed sum until first non-positive run or up to len/2
        m = 1
        while m < len(acf) and acf[m] > 0:
            m += 1
        M = max(1, min(m, len(acf)//2))
    else:
        M = int(cutoff)
    return 0.5 + np.sum(acf[1:M])

# Work with returns rather than raw price
ret = np.diff(np.log(mid))
acf = autocorr(ret)
tau = tau_int(acf)
N = len(ret)
neff_global = N / (2*tau + 1)

# Sliding window Neff
win, step = 200, 20
times_w, neff_w, tau_w = [], [], []
for i in range(0, len(ret)-win, step):
    seg = ret[i:i+win]
    acf_w = autocorr(seg)
    tau_wi = tau_int(acf_w)
    neff_wi = len(seg) / (2*tau_wi + 1)
    tau_w.append(tau_wi)
    neff_w.append(neff_wi)
    times_w.append(t.iloc[i + win])

tau_w = np.array(tau_w); neff_w = np.array(neff_w)

# Plots for Post 2
plt.figure(); plt.plot(acf[:200]); plt.title("Autocorrelation (returns)"); plt.xlabel("Lag"); plt.ylabel("ACF")
plt.savefig(FIG_P2 / "p2_acf_returns.png", bbox_inches="tight")

plt.figure(); plt.plot(times_w, tau_w); plt.title("Integrated autocorr time (sliding)"); plt.xlabel("Time"); plt.ylabel("tau_int")
plt.savefig(FIG_P2 / "p2_tau_int_sliding.png", bbox_inches="tight")

plt.figure(); plt.plot(times_w, neff_w); plt.title("Effective sample size (sliding)"); plt.xlabel("Time"); plt.ylabel("Neff (per window)")
plt.savefig(FIG_P2 / "p2_neff_sliding.png", bbox_inches="tight")

# Project landing supporting figure
plt.figure(); plt.plot(times_w, neff_w)
plt.title("Stationarity & information content (Neff)"); plt.xlabel("Time"); plt.ylabel("Neff")
plt.savefig(FIG_PROJ / "hero_neff.png", bbox_inches="tight")

# %% ---------- Post 3: Committor-like regime probability ----------
# Label future regime by forward return over horizon H
H = 6  # 6*5m = 30 minutes
fwd_ret = np.r_[np.full(H, np.nan), np.log(mid[H:]) - np.log(mid[:-H])]
thr = 0.001  # 0.1% as neutral zone half-width
# y: 1 = up regime, 0 = down regime, nan = neutral
y = np.where(fwd_ret > thr, 1, np.where(fwd_ret < -thr, 0, np.nan))

# Features from past window
def make_features(x, w=24):
    # x is price mid
    x = np.asarray(x)
    # compute SG features again with small window to be fast/adaptive
    sg_loc = savgol_filter(x, 21, 3)
    slope  = savgol_filter(x, 21, 3, deriv=1, delta=dt_sec)
    vol    = pd.Series(np.r_[np.nan, np.diff(np.log(x))]).rolling(12).std().to_numpy()
    # pack as DataFrame
    return pd.DataFrame({
        "sg": sg_loc,
        "slope": slope,
        "vol": vol,
    })

X = make_features(mid)
mask = (~np.isnan(y)) & X.notnull().all(1)
X_train = X[mask].to_numpy()
y_train = y[mask].astype(int)

# Simple logistic regression pipeline
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=500))
]).fit(X_train, y_train)

# Committor-like probability q(x) = P(up | x)
q = np.full(len(mid), np.nan)
q[mask] = clf.predict_proba(X_train)[:,1]

# Quality metric (only for sanity)
try:
    auc = roc_auc_score(y_train, q[mask])
    print("Committor AUC (up vs down):", round(float(auc), 3))
except Exception:
    pass

# Plot probability and highlight transition band near 0.5
plt.figure()
plt.plot(t, q, label="q(t)=P(up)")
plt.axhline(0.5, linestyle="--", linewidth=1)
plt.title("Committor-like probability (logistic regression)"); plt.xlabel("Time"); plt.ylabel("Probability")
plt.savefig(FIG_P3 / "p3_committor_prob.png", bbox_inches="tight")

# Identify transition points ~ on the knife-edge
trans_idx = np.where((q > 0.45) & (q < 0.55))[0]
plt.figure()
plt.plot(t, sg, label="SG price")
plt.scatter(t.iloc[trans_idx], sg[trans_idx], s=16, label="~Transition", alpha=0.7)
plt.title("Potential transition region (q≈0.5)"); plt.xlabel("Time"); plt.ylabel("USDT"); plt.legend()
plt.savefig(FIG_P3 / "p3_transition_regions.png", bbox_inches="tight")

# Project landing supporting figure
plt.figure()
plt.plot(t, q)
plt.title("Regime committor q(t)"); plt.xlabel("Time"); plt.ylabel("P(up)")
plt.savefig(FIG_PROJ / "hero_committor.png", bbox_inches="tight")

# %% Done
print("Saved Post 1 figs ->", FIG_P1)
print("Saved Post 2 figs ->", FIG_P2)
print("Saved Post 3 figs ->", FIG_P3)
print("Saved project figs ->", FIG_PROJ)
