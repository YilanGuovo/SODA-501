############################################
# Time leakage demo + DGP/ACF/PACF demo (Python)
# Seed: 123
# Pipeline dirs: data/raw, data/processed, outputs/figures, outputs/tables
###############################################
####required packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ============================================
# 0) PROJECT DIRECTORY SETUP (pipeline style) same as last week
# ============================================
os.makedirs("data/raw", exist_ok=True)         # raw, unprocessed data
os.makedirs("data/processed", exist_ok=True)   # processed/analysis-ready data
os.makedirs("outputs/figures", exist_ok=True)  # plots
os.makedirs("outputs/tables", exist_ok=True)   # results summaries

# Helper: save figure (PNG) + optionally show
SAVE_DPI = 300
SHOW_FIGURES = False  

def savefig(path: str):
    """Save current figure to outputs/figures and close (or show)."""
    plt.tight_layout()
    plt.savefig(path, dpi=SAVE_DPI)
    if SHOW_FIGURES:
        plt.show()
    plt.close()

# --- 0) Setup
np.random.seed(123) ##keep the random number is same everytime

# ============================================
# 1) Create synthetic daily time series
#    (trend + weekly seasonality + AR(1) noise)
# ============================================
n = 600   ##build fake data example here for time series, set time range as 600 days from 2024,01,01
dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
t = np.arange(1, n + 1)
 
trend = 0.02 * t  ### Remember y = constant + beta * trend + beta2 * seasonality + error
weekly = 1.2 * np.sin(2 * np.pi * t / 7)  ### Here is 7 days as a seasonality

phi = 0.65
eps = np.random.normal(loc=0.0, scale=1.0, size=n)
ar_noise = np.empty(n)
ar_noise[0] = eps[0]     ###errors allowed the time is correlated betwen today and yesterday as an example
for i in range(1, n):
    ar_noise[i] = phi * ar_noise[i - 1] + eps[i]

y = 10 + trend + weekly + ar_noise   #DGP， works as the mechanism for Y.
df = pd.DataFrame({"date": dates, "t": t, "y": y})

# Save raw synthetic data
df.to_csv("data/raw/synthetic_daily_series.csv", index=False)

# --- 2) Visualize the series (save figure)
plt.figure(figsize=(10, 4))
plt.plot(df["date"], df["y"])
plt.title("Synthetic daily time series: trend + weekly seasonality + AR(1) noise")
plt.xlabel("Date")
plt.ylabel("y")
savefig("outputs/figures/01_synthetic_series.png")

############################################
# PART A: Time leakage demo (random split vs time split)
############################################

# --- 3) WRONG evaluation: random train/test split (time leakage)
np.random.seed(123)
test_frac = 0.20
test_n = int(np.floor(n * test_frac))

all_idx = np.arange(n)
test_idx_random = np.random.choice(all_idx, size=test_n, replace=False)
train_idx_random = np.setdiff1d(all_idx, test_idx_random)

y_train_random = df.loc[train_idx_random, "y"].to_numpy()
y_test_random  = df.loc[test_idx_random, "y"].to_numpy()

# Fit ARIMA(1,0,0) on randomly selected training points (conceptually wrong)
fit_random = ARIMA(y_train_random, order=(1, 0, 0)).fit()
pred_random = fit_random.forecast(steps=len(y_test_random))

rmse_random = np.sqrt(np.mean((y_test_random - pred_random) ** 2))

### randomly assigned will lead to mix the "future" variables in the past time, break the time construction.

# --- 4) RIGHT evaluation: train on past, test on future

## Using the past time as train set and predict the future
cut = n - test_n
train_idx_time = np.arange(0, cut)
test_idx_time = np.arange(cut, n)

y_train_time = df.loc[train_idx_time, "y"].to_numpy()
y_test_time  = df.loc[test_idx_time, "y"].to_numpy()

fit_time = ARIMA(y_train_time, order=(1, 0, 0)).fit()
pred_time = fit_time.forecast(steps=len(y_test_time))

rmse_time = np.sqrt(np.mean((y_test_time - pred_time) ** 2))

# Save processed forecast results
forecast_df = pd.DataFrame({
    "date": df.loc[test_idx_time, "date"].to_numpy(),
    "y_true": y_test_time,
    "y_pred": pred_time
})
forecast_df.to_csv("data/processed/forecast_time_split.csv", index=False)

# --- 5) Plot correct evaluation (save figure)
plt.figure(figsize=(10, 4))
plt.plot(df["date"], df["y"], label="Observed y")
plt.axvline(df.loc[cut, "date"], linestyle="--", label="Train/Test cutoff")
plt.plot(df.loc[test_idx_time, "date"], pred_time, label="Forecast (future)")
plt.title("Correct evaluation: train on past, test on future")
plt.xlabel("Date")
plt.ylabel("y")
plt.legend()
savefig("outputs/figures/02_time_split_forecast.png")

############################################
# PART B: Synthetic DGP demo + ACF/PACF diagnostics
############################################

# --- 6) Generate data from known DGP: trend + AR(1) errors
#As a real word example, mainly examines how the current data correlated with past date, specifically at what time.
# DGP: y_t = alpha + delta*t + e_t ; e_t = phi*e_{t-1} + u_t
np.random.seed(123)
n2 = 300
t2 = np.arange(1, n2 + 1)

alpha = 5
delta = 0.03
phi2 = 0.75
u = np.random.normal(loc=0.0, scale=1.0, size=n2)

e = np.empty(n2)
e[0] = u[0]
for i in range(1, n2):
    e[i] = phi2 * e[i - 1] + u[i]

y2 = alpha + delta * t2 + e
df2 = pd.DataFrame({"t": t2, "y2": y2})
df2.to_csv("data/raw/dgp_series.csv", index=False)

# --- 7) Plot the DGP series (save figure)
plt.figure(figsize=(8, 4))
plt.plot(t2, y2)
plt.title("Synthetic DGP: linear trend + AR(1) errors")
plt.xlabel("t")
plt.ylabel("y_t")
savefig("outputs/figures/03_dgp_series.png")

# --- 8) ACF and PACF on y2 (save figures) （Autocorrelation fuction + Direct effect, example: Today, yesterday, and the day before yesterday, PACF can explain today and the day before yesterday)
plt.figure(figsize=(8, 4))
plot_acf(y2, ax=plt.gca(), lags=40)
plt.title("ACF of y_t (trend + AR errors)")
savefig("outputs/figures/04_acf_original.png")

plt.figure(figsize=(8, 4))
plot_pacf(y2, ax=plt.gca(), lags=40, method="ywm")
plt.title("PACF of y_t (trend + AR errors)")
savefig("outputs/figures/05_pacf_original.png")

# --- 9) Detrend via OLS (manual) and re-check ACF/PACF on residuals
#Detrend for some external reasons, such as macro, etc, while we need to check the relatioinship between IV and DV
X = np.column_stack([np.ones(n2), t2])  # intercept + trend
beta_hat = np.linalg.lstsq(X, y2, rcond=None)[0]
y2_hat = X @ beta_hat
resid2 = y2 - y2_hat

resid_df = pd.DataFrame({"t": t2, "resid2": resid2})
resid_df.to_csv("data/processed/detrended_residuals.csv", index=False)

plt.figure(figsize=(8, 4))
plt.plot(t2, resid2)
plt.title("Residuals after removing linear trend")
plt.xlabel("t")
plt.ylabel("residual")
savefig("outputs/figures/06_residuals.png")

plt.figure(figsize=(8, 4))
plot_acf(resid2, ax=plt.gca(), lags=40)
plt.title("ACF of residuals (trend removed)")
savefig("outputs/figures/07_acf_residuals.png")

plt.figure(figsize=(8, 4))
plot_pacf(resid2, ax=plt.gca(), lags=40, method="ywm")
plt.title("PACF of residuals (trend removed)")
savefig("outputs/figures/08_pacf_residuals.png")

# --- 10) Fit AR(1) to residuals and compare phi
fit_ar1 = ARIMA(resid2, order=(1, 0, 0)).fit()

# Robustly grab AR(1) coefficient by name (safer than params[1])
param_names = list(fit_ar1.param_names)
ar1_idx = [i for i, name in enumerate(param_names) if "ar.L1" in name]
phi_hat = float(fit_ar1.params[ar1_idx[0]]) if ar1_idx else float("nan")

# ============================================
# Save results summary (tables)
# ============================================
summary_path = "outputs/tables/results_summary.txt"
with open(summary_path, "w") as f:
    f.write("Time leakage demo (ARIMA(1,0,0))\n")
    f.write("================================\n")
    f.write(f"WRONG (random split) RMSE: {rmse_random:.6f}\n")
    f.write(f"RIGHT (time split)  RMSE: {rmse_time:.6f}\n\n")
    f.write("DGP AR(1) recovery on detrended residuals\n")
    f.write("========================================\n")
    f.write(f"DGP truth phi2: {phi2}\n")
    f.write(f"Estimated phi:  {phi_hat:.6f}\n")

print("\n✅ Pipeline run complete.")
print("Saved:")
print("- data/raw/synthetic_daily_series.csv")
print("- data/processed/forecast_time_split.csv")
print("- data/raw/dgp_series.csv")
print("- data/processed/detrended_residuals.csv")
print("- outputs/figures/01...08_*.png")
print(f"- {summary_path}")
