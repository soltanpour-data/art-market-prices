"""
Step 4 — Hedonic regression (the core of the thesis).

    log(price_i) = X_i * beta + epsilon_i

OLS with heteroscedasticity-robust standard errors (HC1). Annual time
dummies are included so script 05 can build a quality-adjusted price index
from them. Diagnostics: R-squared, variance inflation factors (VIF),
residual plot, and a coefficient plot of the largest price effects.

Output: output/hedonic_regression_summary.txt,
        output/hedonic_coefficients.csv,
        output/fig_residuals.png, output/fig_coefficients.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv("data/model_data.csv")

meta = ["artist", "subject", "auction_house", "sale_year", "sale_month",
        "log_price", "price_gbp", "year_trend"]
X = df.drop(columns=meta)
y = df["log_price"]
X = sm.add_constant(X, has_constant="add").astype(float)

model = sm.OLS(y, X).fit(cov_type="HC1")   # robust standard errors
print(model.summary())

with open("output/hedonic_regression_summary.txt", "w") as f:
    f.write(str(model.summary()))

coefs = pd.DataFrame({
    "variable": model.params.index,
    "coef": model.params.values,
    "std_err": model.bse.values,
    "p_value": model.pvalues.values,
}).sort_values("p_value")
coefs.to_csv("output/hedonic_coefficients.csv", index=False)

print(f"\nR-squared: {model.rsquared:.3f} | adj. R-squared: {model.rsquared_adj:.3f}")

# VIF on the house + subject dummies (no exact nesting is expected there)
core = [c for c in X.columns
        if c.startswith(("house_", "subj_")) and c in X.columns]
vif = pd.DataFrame({"variable": core,
                    "VIF": [variance_inflation_factor(X[core].values, i)
                            for i in range(len(core))]})
print("\nVIF (values > 10 signal multicollinearity):")
print(vif.sort_values("VIF", ascending=False).head(10).to_string(index=False))

# Residuals vs fitted (subsample for speed)
samp = model.fittedvalues.sample(n=min(20000, len(df)), random_state=7)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(samp, model.resid.loc[samp.index], s=4, alpha=0.2)
ax.axhline(0, color="red", lw=1)
ax.set_title("Residuals vs fitted (hedonic OLS)")
ax.set_xlabel("Fitted log price"); ax.set_ylabel("Residuals")
fig.tight_layout(); fig.savefig("output/fig_residuals.png"); plt.close(fig)

# Coefficient plot: 15 largest non-time effects with 95% CI
plot_df = coefs[~coefs["variable"].str.startswith(("y_", "const"))].copy()
plot_df["abs_coef"] = plot_df["coef"].abs()
plot_df = plot_df.nlargest(15, "abs_coef").sort_values("coef")
y_pos = range(len(plot_df))
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(plot_df["coef"], y_pos,
            xerr=1.96 * plot_df["std_err"], fmt="o", capsize=3)
ax.set_yticks(list(y_pos)); ax.set_yticklabels(plot_df["variable"], fontsize=8)
ax.axvline(0, color="red", lw=1)
ax.set_title("Hedonic regression: largest price effects (95% CI)")
ax.set_xlabel("Coefficient (change in log price)")
fig.tight_layout(); fig.savefig("output/fig_coefficients.png"); plt.close(fig)

print("\nSaved summary, coefficients and plots to output/")
