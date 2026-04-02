<<<<<<< HEAD
############################################
# Time leakage demo + DGP/ACF/PACF demo
# + EX 3 Decomposition
# + EX 4 Rolling-origin backtest
# + EX 5 Interrupted Time Series
# Seed: 123
############################################

# --- 0) Setup
set.seed(123)

use_forecast <- requireNamespace("forecast", quietly = TRUE)

dir.create("outputs", showWarnings = FALSE)
dir.create("outputs/figures", showWarnings = FALSE, recursive = TRUE)
dir.create("outputs/tables", showWarnings = FALSE, recursive = TRUE)

############################################
# --- 1) Create synthetic daily time series
############################################

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

############################################
# PART A: Time leakage demo
############################################

set.seed(123)
test_frac <- 0.20
test_n <- floor(n * test_frac)

# WRONG random split
test_idx_random <- sample(1:n, size = test_n, replace = FALSE)
train_idx_random <- setdiff(1:n, test_idx_random)

y_train_random <- df$y[train_idx_random]
y_test_random  <- df$y[test_idx_random]

if (use_forecast) {
  fit_random <- forecast::auto.arima(y_train_random)
  pred_random <- as.numeric(forecast::forecast(fit_random, h = length(y_test_random))$mean)
} else {
  fit_random <- arima(y_train_random, order = c(1,0,0))
  pred_random <- as.numeric(predict(fit_random, n.ahead = length(y_test_random))$pred)
}

rmse_random <- sqrt(mean((y_test_random - pred_random)^2))

cat("\nWRONG Random split RMSE:", rmse_random, "\n")

# RIGHT time split
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

cat("RIGHT Time split RMSE:", rmse_time, "\n")

############################################
# EX 3) Decomposition
############################################

y_ts <- ts(df$y, frequency = 7)
decomp <- stl(y_ts, s.window = "periodic")

png("outputs/figures/decomposition.png", width = 1000, height = 700)
plot(decomp, main = "STL decomposition (weekly seasonality)")
dev.off()

############################################
# EX 4) Rolling-origin backtesting
############################################

h <- 1
init_window <- 300

bt_t <- init_window:(n - h)

yhat_bt <- rep(NA_real_, n)
err_bt  <- rep(NA_real_, n)

for (tt in bt_t) {
  y_train <- df$y[1:tt]
  fit_ar1_tt <- arima(y_train, order = c(1,0,0))
  pred_tt1 <- as.numeric(predict(fit_ar1_tt, n.ahead = 1)$pred)
  yhat_bt[tt + 1] <- pred_tt1
  err_bt[tt + 1]  <- df$y[tt + 1] - pred_tt1
}

bt_idx <- (init_window + 1):n

bt_df <- data.frame(
  date  = df$date[bt_idx],
  y     = df$y[bt_idx],
  yhat  = yhat_bt[bt_idx],
  error = err_bt[bt_idx]
)

rmse_backtest <- sqrt(mean(bt_df$error^2, na.rm = TRUE))

cat("Rolling-origin RMSE:", rmse_backtest, "\n")

write.csv(bt_df, "outputs/tables/backtest_errors.csv", row.names = FALSE)

png("outputs/figures/backtest_forecast.png", width = 1100, height = 600)
plot(bt_df$date, bt_df$y, type = "l",
     main = "Rolling-origin backtest: observed vs forecast",
     xlab = "Date", ylab = "y")
lines(bt_df$date, bt_df$yhat)
legend("topleft",
       legend = c("Observed y", "Forecast yhat"),
       lty = c(1,1),
       bty = "n")
dev.off()

############################################
# EX 5) Interrupted Time Series (ITS)
############################################

t0 <- 300
cat("Intervention index t0 =", t0, "\n")

I0 <- as.numeric(df$t >= t0)
post0 <- (df$t - t0) * I0

# synthetic intervention
tau1_true <- 2.0
tau2_true <- 0.01

y_its <- df$y + tau1_true * I0 + tau2_true * post0

its_fit <- lm(y_its ~ df$t + I0 + post0)

yhat_its <- fitted(its_fit)

coefs <- coef(its_fit)
alpha_hat <- unname(coefs[1])
delta_hat <- unname(coefs[2])
y_cf <- alpha_hat + delta_hat * df$t

# placebo
t0_placebo <- 200
I_p <- as.numeric(df$t >= t0_placebo)
post_p <- (df$t - t0_placebo) * I_p

its_placebo_fit <- lm(y_its ~ df$t + I_p + post_p)

summ_real <- summary(its_fit)$coefficients
summ_pl   <- summary(its_placebo_fit)$coefficients

real_df <- data.frame(
  model = "real",
  term = rownames(summ_real),
  estimate = summ_real[,1],
  std_error = summ_real[,2],
  t_value = summ_real[,3],
  p_value = summ_real[,4],
  row.names = NULL
)

placebo_df <- data.frame(
  model = "placebo",
  term = rownames(summ_pl),
  estimate = summ_pl[,1],
  std_error = summ_pl[,2],
  t_value = summ_pl[,3],
  p_value = summ_pl[,4],
  row.names = NULL
)

its_results <- rbind(real_df, placebo_df)

write.csv(its_results, "outputs/tables/its_results.csv", row.names = FALSE)

png("outputs/figures/its_plot.png", width = 1100, height = 650)
plot(df$date, y_its, type = "l",
     main = "Interrupted Time Series",
     xlab = "Date", ylab = "y")
lines(df$date, yhat_its)
lines(df$date, y_cf, lty = 2)
abline(v = df$date[t0], lty = 3)
legend("topleft",
       legend = c("Observed", "ITS fitted", "Counterfactual", "Intervention"),
       lty = c(1,1,2,3),
       bty = "n")
dev.off()

cat("All outputs saved in outputs/ folder.\n")
=======
############################################
# Time leakage demo + DGP/ACF/PACF demo
# + EX 3 Decomposition
# + EX 4 Rolling-origin backtest
# + EX 5 Interrupted Time Series
# Seed: 123
############################################

# --- 0) Setup
set.seed(123)

use_forecast <- requireNamespace("forecast", quietly = TRUE)

dir.create("outputs", showWarnings = FALSE)
dir.create("outputs/figures", showWarnings = FALSE, recursive = TRUE)
dir.create("outputs/tables", showWarnings = FALSE, recursive = TRUE)

############################################
# --- 1) Create synthetic daily time series
############################################

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

############################################
# PART A: Time leakage demo
############################################

set.seed(123)
test_frac <- 0.20
test_n <- floor(n * test_frac)

# WRONG random split
test_idx_random <- sample(1:n, size = test_n, replace = FALSE)
train_idx_random <- setdiff(1:n, test_idx_random)

y_train_random <- df$y[train_idx_random]
y_test_random  <- df$y[test_idx_random]

if (use_forecast) {
  fit_random <- forecast::auto.arima(y_train_random)
  pred_random <- as.numeric(forecast::forecast(fit_random, h = length(y_test_random))$mean)
} else {
  fit_random <- arima(y_train_random, order = c(1,0,0))
  pred_random <- as.numeric(predict(fit_random, n.ahead = length(y_test_random))$pred)
}

rmse_random <- sqrt(mean((y_test_random - pred_random)^2))

cat("\nWRONG Random split RMSE:", rmse_random, "\n")

# RIGHT time split
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

cat("RIGHT Time split RMSE:", rmse_time, "\n")

############################################
# EX 3) Decomposition
############################################

y_ts <- ts(df$y, frequency = 7)
decomp <- stl(y_ts, s.window = "periodic")

png("outputs/figures/decomposition.png", width = 1000, height = 700)
plot(decomp, main = "STL decomposition (weekly seasonality)")
dev.off()

############################################
# EX 4) Rolling-origin backtesting
############################################

h <- 1
init_window <- 300

bt_t <- init_window:(n - h)

yhat_bt <- rep(NA_real_, n)
err_bt  <- rep(NA_real_, n)

for (tt in bt_t) {
  y_train <- df$y[1:tt]
  fit_ar1_tt <- arima(y_train, order = c(1,0,0))
  pred_tt1 <- as.numeric(predict(fit_ar1_tt, n.ahead = 1)$pred)
  yhat_bt[tt + 1] <- pred_tt1
  err_bt[tt + 1]  <- df$y[tt + 1] - pred_tt1
}

bt_idx <- (init_window + 1):n

bt_df <- data.frame(
  date  = df$date[bt_idx],
  y     = df$y[bt_idx],
  yhat  = yhat_bt[bt_idx],
  error = err_bt[bt_idx]
)

rmse_backtest <- sqrt(mean(bt_df$error^2, na.rm = TRUE))

cat("Rolling-origin RMSE:", rmse_backtest, "\n")

write.csv(bt_df, "outputs/tables/backtest_errors.csv", row.names = FALSE)

png("outputs/figures/backtest_forecast.png", width = 1100, height = 600)
plot(bt_df$date, bt_df$y, type = "l",
     main = "Rolling-origin backtest: observed vs forecast",
     xlab = "Date", ylab = "y")
lines(bt_df$date, bt_df$yhat)
legend("topleft",
       legend = c("Observed y", "Forecast yhat"),
       lty = c(1,1),
       bty = "n")
dev.off()

############################################
# EX 5) Interrupted Time Series (ITS)
############################################

t0 <- 300
cat("Intervention index t0 =", t0, "\n")

I0 <- as.numeric(df$t >= t0)
post0 <- (df$t - t0) * I0

# synthetic intervention
tau1_true <- 2.0
tau2_true <- 0.01

y_its <- df$y + tau1_true * I0 + tau2_true * post0

its_fit <- lm(y_its ~ df$t + I0 + post0)

yhat_its <- fitted(its_fit)

coefs <- coef(its_fit)
alpha_hat <- unname(coefs[1])
delta_hat <- unname(coefs[2])
y_cf <- alpha_hat + delta_hat * df$t

# placebo
t0_placebo <- 200
I_p <- as.numeric(df$t >= t0_placebo)
post_p <- (df$t - t0_placebo) * I_p

its_placebo_fit <- lm(y_its ~ df$t + I_p + post_p)

summ_real <- summary(its_fit)$coefficients
summ_pl   <- summary(its_placebo_fit)$coefficients

real_df <- data.frame(
  model = "real",
  term = rownames(summ_real),
  estimate = summ_real[,1],
  std_error = summ_real[,2],
  t_value = summ_real[,3],
  p_value = summ_real[,4],
  row.names = NULL
)

placebo_df <- data.frame(
  model = "placebo",
  term = rownames(summ_pl),
  estimate = summ_pl[,1],
  std_error = summ_pl[,2],
  t_value = summ_pl[,3],
  p_value = summ_pl[,4],
  row.names = NULL
)

its_results <- rbind(real_df, placebo_df)

write.csv(its_results, "outputs/tables/its_results.csv", row.names = FALSE)

png("outputs/figures/its_plot.png", width = 1100, height = 650)
plot(df$date, y_its, type = "l",
     main = "Interrupted Time Series",
     xlab = "Date", ylab = "y")
lines(df$date, yhat_its)
lines(df$date, y_cf, lty = 2)
abline(v = df$date[t0], lty = 3)
legend("topleft",
       legend = c("Observed", "ITS fitted", "Counterfactual", "Intervention"),
       lty = c(1,1,2,3),
       bty = "n")
dev.off()

cat("All outputs saved in outputs/ folder.\n")
>>>>>>> d9681faaefa0c817f85f304ca745e7ad9ee5a352
