###############################################################################
# Measurement Error + Placebo Tests Tutorial: Python
# Author: Jared Edgerton
#
# Adapted for assignment submission with minimal changes:
#   - safer working directory setup
#   - clean tables for Question 3
#   - validation-share comparison for Question 4
#   - summary outputs for Question 5
###############################################################################

# %%
# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use current script folder as working directory (safer than hard-coded local path)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
print(os.getcwd())

# Reproducibility
np.random.seed(123)

# Create common project folders (safe to run repeatedly)
os.makedirs("data_raw", exist_ok=True)
os.makedirs("data_processed", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -----------------------------------------------------------------------------
# Part 1: A simple data generating process with confounding
# -----------------------------------------------------------------------------
n = 5000

# True confounder
x_true = np.random.normal(loc=0.0, scale=1.0, size=n)

# Treatment assignment correlated with x_true (logistic link)
logit_p = 1.0 * x_true
p = 1.0 / (1.0 + np.exp(-logit_p))
d = np.random.binomial(n=1, p=p, size=n)

# True outcome model
tau = 1.0     # true effect of D on Y
beta = 1.0    # effect of X_true on Y
eps_y = np.random.normal(loc=0.0, scale=1.0, size=n)

y = tau * d + beta * x_true + eps_y

# Placebo outcome (negative control outcome): NOT affected by D by construction
eps_pl = np.random.normal(loc=0.0, scale=1.0, size=n)
y_placebo = 0.0 * d + beta * x_true + eps_pl

df_base = pd.DataFrame(
    {
        "y": y,
        "y_placebo": y_placebo,
        "d": d,
        "x_true": x_true,
    }
)

print("\n--- Data preview ---")
print(df_base.head())
print("\nTreatment rate:", round(df_base["d"].mean(), 4))

# -----------------------------------------------------------------------------
# Part 2: Measurement error settings
# -----------------------------------------------------------------------------
sigma_u_grid = [0.0, 0.2, 0.5, 1.0, 2.0]
R = 30  # repetitions (to see variability from measurement error draws)

# Fix a "validation sample" index set (20% of observations)
validation_share = 0.20
validation_idx = np.random.choice(np.arange(n), size=int(validation_share * n), replace=False)
is_validation = np.zeros(n, dtype=bool)
is_validation[validation_idx] = True

# Storage
rows = []

print("\n--- Running measurement error simulations ---")
for sigma_u in sigma_u_grid:
    tau_oracle_list = []
    tau_naive_list = []
    tau_cal_list = []
    tau_placebo_list = []

    beta_oracle_list = []
    beta_naive_list = []
    beta_cal_list = []

    for r in range(R):
        # Draw measurement error and observed covariate
        u = np.random.normal(loc=0.0, scale=sigma_u, size=n)
        x_obs = x_true + u

        # Build design matrices with intercept column
        ones = np.ones(n)

        # (A) Oracle regression: y ~ 1 + d + x_true
        X_oracle = np.column_stack([ones, d, x_true])
        coef_oracle, _, _, _ = np.linalg.lstsq(X_oracle, y, rcond=None)
        tau_oracle_list.append(coef_oracle[1])
        beta_oracle_list.append(coef_oracle[2])

        # (B) Naive regression: y ~ 1 + d + x_obs
        X_naive = np.column_stack([ones, d, x_obs])
        coef_naive, _, _, _ = np.linalg.lstsq(X_naive, y, rcond=None)
        tau_naive_list.append(coef_naive[1])
        beta_naive_list.append(coef_naive[2])

        # (C) Regression calibration (validation subsample)
        ones_val = np.ones(int(validation_share * n))
        X_cal_val = np.column_stack([ones_val, x_obs[is_validation]])
        coef_cal, _, _, _ = np.linalg.lstsq(X_cal_val, x_true[is_validation], rcond=None)
        x_hat = coef_cal[0] + coef_cal[1] * x_obs

        X_calibrated = np.column_stack([ones, d, x_hat])
        coef_calibrated, _, _, _ = np.linalg.lstsq(X_calibrated, y, rcond=None)
        tau_cal_list.append(coef_calibrated[1])
        beta_cal_list.append(coef_calibrated[2])

        # Outcome placebo: y_placebo ~ 1 + d + x_obs
        X_placebo = np.column_stack([ones, d, x_obs])
        coef_placebo, _, _, _ = np.linalg.lstsq(X_placebo, y_placebo, rcond=None)
        tau_placebo_list.append(coef_placebo[1])

    # Summaries per sigma_u
    rows.append(
        {
            "sigma_u": sigma_u,
            "tau_true": tau,
            "tau_oracle_mean": float(np.mean(tau_oracle_list)),
            "tau_naive_mean": float(np.mean(tau_naive_list)),
            "tau_cal_mean": float(np.mean(tau_cal_list)),
            "tau_placebo_mean": float(np.mean(tau_placebo_list)),
            "tau_oracle_q025": float(np.quantile(tau_oracle_list, 0.025)),
            "tau_oracle_q975": float(np.quantile(tau_oracle_list, 0.975)),
            "tau_naive_q025": float(np.quantile(tau_naive_list, 0.025)),
            "tau_naive_q975": float(np.quantile(tau_naive_list, 0.975)),
            "tau_cal_q025": float(np.quantile(tau_cal_list, 0.025)),
            "tau_cal_q975": float(np.quantile(tau_cal_list, 0.975)),
            "beta_true": beta,
            "beta_oracle_mean": float(np.mean(beta_oracle_list)),
            "beta_naive_mean": float(np.mean(beta_naive_list)),
            "beta_cal_mean": float(np.mean(beta_cal_list)),
        }
    )

    print(f"  done sigma_u={sigma_u}")

results = pd.DataFrame(rows)
print("\n--- Summary (means) ---")
print(results[["sigma_u", "tau_true", "tau_oracle_mean", "tau_naive_mean", "tau_cal_mean", "tau_placebo_mean"]].to_string(index=False))

results.to_csv("outputs/measurement_error_results.csv", index=False)

# Clean table for Question 3
q3_clean_table = results[[
    "sigma_u",
    "tau_oracle_mean", "tau_naive_mean", "tau_cal_mean",
    "beta_oracle_mean", "beta_naive_mean", "beta_cal_mean"
]].copy()
q3_clean_table = q3_clean_table.round(4)
q3_clean_table.to_csv("outputs/q3_clean_table.csv", index=False)
print("\n--- Q3 clean table ---")
print(q3_clean_table.to_string(index=False))

# -----------------------------------------------------------------------------
# Part 3: Plot how measurement error changes estimates
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(results["sigma_u"], results["tau_oracle_mean"],
         marker="o", color="blue", linewidth=2,
         label="Oracle: y ~ d + x_true")

plt.plot(results["sigma_u"], results["tau_naive_mean"],
         marker="o", color="orange", linewidth=3,
         linestyle="--",
         label="Naive: y ~ d + x_obs")

plt.plot(results["sigma_u"], results["tau_cal_mean"],
         marker="o", color="green", linewidth=2,
         label="Calibration: y ~ d + x_hat")

plt.plot(results["sigma_u"], results["tau_placebo_mean"],
         marker="s", color="red", linewidth=2,
         linestyle=":",
         label="Outcome placebo: y_pl ~ d + x_obs")

plt.axhline(tau, linestyle="--", color="black", label="True tau")

plt.title("Estimated treatment effect vs measurement error in confounder")
plt.xlabel("Measurement error SD (sigma_u)")
plt.ylabel("Estimated coefficient on d")
plt.legend()
plt.tight_layout()
plt.savefig("figures/measurement_error_tau_vs_sigma.png", dpi=200)
plt.close()

# Plot beta estimates vs sigma_u (confounder coefficient attenuation)
plt.figure(figsize=(8, 5))
plt.plot(results["sigma_u"], results["beta_oracle_mean"], marker="o", label="Oracle: coef on x_true")
plt.plot(results["sigma_u"], results["beta_naive_mean"], marker="o", label="Naive: coef on x_obs")
plt.plot(results["sigma_u"], results["beta_cal_mean"], marker="o", label="Calibration: coef on x_hat")
plt.axhline(beta, linestyle="--", label="True beta")
plt.title("Estimated confounder effect vs measurement error (attenuation)")
plt.xlabel("Measurement error SD (sigma_u)")
plt.ylabel("Estimated coefficient on confounder term")
plt.legend()
plt.tight_layout()
plt.savefig("figures/measurement_error_beta_vs_sigma.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# Part 4: Validation subsample and regression calibration
# -----------------------------------------------------------------------------
validation_share_grid = [0.05, 0.20, 0.50]
sigma_u_fixed = 1.0
validation_rows = []

print("\n--- Running validation-share comparison (sigma_u fixed at 1.0) ---")
for validation_share_loop in validation_share_grid:
    tau_naive_list = []
    tau_cal_list = []

    validation_idx_loop = np.random.choice(
        np.arange(n), size=int(validation_share_loop * n), replace=False
    )
    is_validation_loop = np.zeros(n, dtype=bool)
    is_validation_loop[validation_idx_loop] = True

    for r in range(R):
        u = np.random.normal(loc=0.0, scale=sigma_u_fixed, size=n)
        x_obs = x_true + u
        ones = np.ones(n)

        # Naive regression
        X_naive = np.column_stack([ones, d, x_obs])
        coef_naive, _, _, _ = np.linalg.lstsq(X_naive, y, rcond=None)
        tau_naive_list.append(coef_naive[1])

        # Calibration regression using chosen validation share
        ones_val = np.ones(int(validation_share_loop * n))
        X_cal_val = np.column_stack([ones_val, x_obs[is_validation_loop]])
        coef_cal, _, _, _ = np.linalg.lstsq(X_cal_val, x_true[is_validation_loop], rcond=None)
        x_hat = coef_cal[0] + coef_cal[1] * x_obs

        X_calibrated = np.column_stack([ones, d, x_hat])
        coef_calibrated, _, _, _ = np.linalg.lstsq(X_calibrated, y, rcond=None)
        tau_cal_list.append(coef_calibrated[1])

    validation_rows.append(
        {
            "validation_share": validation_share_loop,
            "sigma_u": sigma_u_fixed,
            "tau_naive_mean": float(np.mean(tau_naive_list)),
            "tau_cal_mean": float(np.mean(tau_cal_list)),
            "cal_minus_naive": float(np.mean(tau_cal_list) - np.mean(tau_naive_list)),
        }
    )
    print(f"  done validation_share={validation_share_loop}")

validation_results = pd.DataFrame(validation_rows).round(4)
validation_results.to_csv("outputs/q4_validation_share_comparison.csv", index=False)
print("\n--- Q4 validation-share comparison ---")
print(validation_results.to_string(index=False))

# -----------------------------------------------------------------------------
# Part 5: Treatment permutation placebo (randomization inference)
# -----------------------------------------------------------------------------
sigma_u_perm = 1.0
u_perm = np.random.normal(loc=0.0, scale=sigma_u_perm, size=n)
x_obs_perm = x_true + u_perm

ones = np.ones(n)
X_obs = np.column_stack([ones, d, x_obs_perm])
coef_obs, _, _, _ = np.linalg.lstsq(X_obs, y, rcond=None)
tau_hat_obs = float(coef_obs[1])

print("\n--- Permutation placebo setup ---")
print("sigma_u used:", sigma_u_perm)
print("Observed tau_hat (naive model):", round(tau_hat_obs, 4))

B = 500
tau_perm = []

for b in range(B):
    d_perm = np.random.permutation(d)
    X_b = np.column_stack([ones, d_perm, x_obs_perm])
    coef_b, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
    tau_perm.append(float(coef_b[1]))

tau_perm = np.array(tau_perm)

# Empirical two-sided p-value
p_emp = (1.0 + np.sum(np.abs(tau_perm) >= np.abs(tau_hat_obs))) / (B + 1.0)
print("Empirical p-value (two-sided):", round(p_emp, 4))

perm_df = pd.DataFrame({"tau_perm": tau_perm})
perm_df.to_csv("outputs/permutation_tau_distribution.csv", index=False)

# Q5 summary output
outcome_placebo_sigma1 = results.loc[results["sigma_u"] == 1.0, "tau_placebo_mean"].iloc[0]
q5_summary = pd.DataFrame([
    {
        "sigma_u_outcome_placebo": 1.0,
        "outcome_placebo_coef_on_d": float(outcome_placebo_sigma1),
        "sigma_u_permutation": sigma_u_perm,
        "tau_hat_obs": float(tau_hat_obs),
        "empirical_two_sided_p_value": float(p_emp),
        "num_permutations_B": B,
    }
]).round(4)
q5_summary.to_csv("outputs/q5_placebo_summary.csv", index=False)
print("\n--- Q5 placebo summary ---")
print(q5_summary.to_string(index=False))

# Plot permutation distribution + observed line
plt.figure(figsize=(8, 5))
plt.hist(tau_perm, bins=30, alpha=0.8)
plt.axvline(tau_hat_obs, linestyle="--", linewidth=2, label=f"Observed tau_hat = {tau_hat_obs:.3f}")
plt.axvline(-tau_hat_obs, linestyle="--", linewidth=1)
plt.title(f"Treatment permutation placebo (sigma_u={sigma_u_perm})\nEmpirical p-value = {p_emp:.3f}")
plt.xlabel("Coefficient on permuted treatment")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("figures/permutation_placebo_tau_hist.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# End of script
# -----------------------------------------------------------------------------
print("\nDone. Outputs written to:")
print("  outputs/measurement_error_results.csv")
print("  outputs/q3_clean_table.csv")
print("  outputs/q4_validation_share_comparison.csv")
print("  outputs/q5_placebo_summary.csv")
print("  outputs/permutation_tau_distribution.csv")
print("  figures/measurement_error_tau_vs_sigma.png")
print("  figures/measurement_error_beta_vs_sigma.png")
print("  figures/permutation_placebo_tau_hist.png")
###############################################################################
