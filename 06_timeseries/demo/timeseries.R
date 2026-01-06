############################################
# Time leakage demo + DGP/ACF/PACF demo in R
# Seed: 123
# Fully sequential, hard-coded, no user-defined functions
############################################

# --- 0) Setup
set.seed(123)

# Optional package check for auto.arima (nice-to-have)
use_forecast <- requireNamespace("forecast", quietly = TRUE)

# --- 1) Create a synthetic daily time series (trend + weekly seasonality + AR(1) noise)
n <- 600
dates <- seq.Date(from = as.Date("2024-01-01"), by = "day", length.out = n)
t <- 1:n

trend <- 0.02 * t
weekly <- 1.2 * sin(2 * pi * t / 7)

phi <- 0.65
eps <- rnorm(n, mean = 0, sd = 1.0)
ar_noise <- rep(NA_real_, n)
ar_noise[1] <- eps[1]
for (i in 2:n) {
  ar_noise[i] <- phi * ar_noise[i - 1] + eps[i]
}

y <- 10 + trend + weekly + ar_noise

df <- data.frame(
  date = dates,
  t = t,
  y = y
)

# --- 2) Visualize the series
plot(df$date, df$y, type = "l",
     main = "Synthetic daily time series: trend + weekly seasonality + AR(1) noise",
     xlab = "Date", ylab = "y")

############################################
# PART A: Time leakage demo (random split vs time split)
############################################

# --- 3) WRONG evaluation: random train/test split (time leakage)
set.seed(123)
test_frac <- 0.20
test_n <- floor(n * test_frac)

test_idx_random <- sample(1:n, size = test_n, replace = FALSE)
train_idx_random <- setdiff(1:n, test_idx_random)

y_train_random <- df$y[train_idx_random]
y_test_random  <- df$y[test_idx_random]

# Fit model on randomly selected training points (conceptually wrong for time series)
if (use_forecast) {
  fit_random <- forecast::auto.arima(y_train_random)
  pred_random <- as.numeric(forecast::forecast(fit_random, h = length(y_test_random))$mean)
} else {
  fit_random <- arima(y_train_random, order = c(1,0,0))
  pred_random <- as.numeric(predict(fit_random, n.ahead = length(y_test_random))$pred)
}

rmse_random <- sqrt(mean((y_test_random - pred_random)^2))

cat("\n==============================\n")
cat("WRONG: Random split RMSE (time leakage): ", rmse_random, "\n")
cat("==============================\n")

# --- 4) RIGHT evaluation: train on past, test on future
cut <- n - test_n
train_idx_time <- 1:cut
test_idx_time  <- (cut + 1):n

y_train_time <- df$y[train_idx_time]
y_test_time  <- df$y[test_idx_time]

if (use_forecast) {
  fit_time <- forecast::auto.arima(y_train_time)
  pred_time <- as.numeric(forecast::forecast(fit_time, h = length(y_test_time))$mean)
} else {
  fit_time <- arima(y_train_time, order = c(1,0,0))
  pred_time <- as.numeric(predict(fit_time, n.ahead = length(y_test_time))$pred)
}

rmse_time <- sqrt(mean((y_test_time - pred_time)^2))

cat("\n==============================\n")
cat("RIGHT: Time split RMSE (train past, test future): ", rmse_time, "\n")
cat("==============================\n")

# --- 5) Plot the correct evaluation (train vs test and forecast)
plot(df$date, df$y, type = "l",
     main = "Correct evaluation: train on past, test on future",
     xlab = "Date", ylab = "y")

abline(v = df$date[cut], lty = 2)
text(df$date[cut], max(df$y), labels = "cutoff", pos = 4)

lines(df$date[test_idx_time], pred_time, lty = 1)

legend("topleft",
       legend = c("Observed y", "Forecast (future)", "Train/Test cutoff"),
       lty = c(1, 1, 2),
       bty = "n")

############################################
# PART B: Synthetic DGP demo (autocorrelation + trend) + ACF/PACF diagnostics
############################################

# --- 6) Generate data from a known DGP: trend + AR(1) errors
# DGP: y_t = alpha + delta*t + e_t ; e_t = phi*e_{t-1} + u_t
set.seed(123)
n2 <- 300
t2 <- 1:n2

alpha <- 5
delta <- 0.03
phi2 <- 0.75
u <- rnorm(n2, mean = 0, sd = 1)

e <- rep(NA_real_, n2)
e[1] <- u[1]
for (i in 2:n2) {
  e[i] <- phi2 * e[i - 1] + u[i]
}

y2 <- alpha + delta * t2 + e

# --- 7) Plot the DGP series
plot(t2, y2, type = "l",
     main = "Synthetic DGP: linear trend + AR(1) errors",
     xlab = "t", ylab = "y_t")

# --- 8) Diagnose dependence with ACF and PACF
# ACF: shows correlation at different lags
acf(y2, main = "ACF of y_t (trend + AR errors)")

# PACF: highlights the 'direct' lag structure
pacf(y2, main = "PACF of y_t (trend + AR errors)")

# --- 9) Detrend and re-check ACF/PACF on residuals
# This separates "trend" from "dependence in errors"
fit_lm <- lm(y2 ~ t2)
resid2 <- residuals(fit_lm)

plot(t2, resid2, type = "l",
     main = "Residuals after removing linear trend (should still show AR structure)",
     xlab = "t", ylab = "residual")

acf(resid2, main = "ACF of residuals (trend removed)")
pacf(resid2, main = "PACF of residuals (trend removed)")

# --- 10) Fit an AR(1) model to residuals and compare estimated phi to truth
fit_ar1 <- arima(resid2, order = c(1,0,0))
cat("\n==============================\n")
cat("DGP truth phi2 = ", phi2, "\n")
cat("Estimated AR(1) phi from residuals = ", fit_ar1$coef[1], "\n")
cat("==============================\n")

# --- 11) Narration-ready takeaway
cat("\nNarration-ready takeaway:\n")
cat("- In the DGP, we *know* the errors are AR(1), so observations are dependent over time.\n")
cat("- ACF/PACF make that dependence visible.\n")
cat("- Removing trend helps isolate autocorrelation in the error process.\n")
cat("- Separately: random splits leak time and look too good; past->future splits are the honest default.\n")
