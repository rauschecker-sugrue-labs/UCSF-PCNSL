# pcnsl_analysis.py — Design Specification

## Overview

A new module `pcnsl_analysis.py` providing three opinionated analysis frameworks for the UCSF-PCNSL dataset. All statistical backends run in R via `rpy2`, using `mgcv` for GAMs, and `survival`/`glmnet` for survival analysis.

The module is **loosely coupled** to `pcnsl_data_loader.py` — all functions accept pandas DataFrames and column name arguments. No internal data loading.

## Architecture

### Shared R Bridge Base

An `RModelBase` abstract base class handles all `rpy2` interop:

- **`_ensure_r_packages()`** — module-level function, runs once on first use. Checks that R and `rpy2` are available; checks that required R packages (`mgcv`, `survival`, `glmnet`) are installed. Raises `ImportError` with actionable install instructions if anything is missing. Offers to install missing R packages via `utils.install.packages()`.
- **`_pandas_to_r(df)`** — converts a pandas DataFrame to an R `data.frame`, handling NA, categorical, and numeric types correctly.
- **`_r_to_pandas(r_obj)`** — converts R data.frames, matrices, and named vectors back to pandas objects.
- **`_build_formula(target, predictors, smooth_terms, smooth_k)`** — constructs an R formula string. Linear terms are added directly; smooth terms are wrapped in `s(term, k=k)`. Returns an `robjects.Formula`.

**`RModelBase`** provides:
- `__init__()` — activates `rpy2`, calls `_ensure_r_packages()`, loads R packages into the instance's R namespace.
- `fit()` — abstract, overridden by subclasses.
- `predict(new_data)` — abstract, overridden by subclasses.
- `summary()` — abstract, overridden by subclasses.
- `self.model_` — stores the fitted R object after `fit()`.
- `self.results_` — stores a Python-side summary dict/DataFrame after `fit()`.

Lazy initialization: `import pcnsl_analysis` does **not** require R. The R bridge activates only when a class is instantiated.

---

## Framework 1: GAMModel

**Purpose:** General-purpose generalized additive model wrapper for predicting any continuous (or other family) response variable. Uses `mgcv::gam()`.

### API

```python
class GAMModel(RModelBase):
    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        predictors: list[str],
        smooth_terms: list[str] = None,
        smooth_k: int | dict = None,
        family: str = "gaussian",
        method: str = "REML",
    ) -> "GAMModel": ...

    def predict(
        self,
        new_data: pd.DataFrame = None,
        type: str = "response",
        se_fit: bool = True,
    ) -> pd.DataFrame: ...

    def summary(self) -> dict: ...
    def plot_smooth_terms(self, terms: list[str] = None, **kwargs): ...
    def check_diagnostics(self): ...
    def anova(self, other: "GAMModel" = None) -> pd.DataFrame: ...
```

### Key Behaviors

- **Formula construction:** `target ~ x1 + x2 + s(x3, k=10) + s(x4, k=6)`. `predictors` are linear terms; `smooth_terms` get wrapped in `s()`. The user explicitly decides which covariates are nonlinear.
- **`family`** is a string mapping directly to R family objects. Supports all `mgcv` families including `tw()` (Tweedie, useful for zero-inflated lesion volumes) and `nb()` (negative binomial).
- **`method`** controls smoothness selection: `REML` (default, recommended), `GCV.Cp`, `ML`.
- **`smooth_k`** can be an int (applied to all smooth terms) or a dict mapping column names to individual k values.
- **`summary()`** returns a dict with keys: `parametric_coefficients` (DataFrame), `smooth_terms` (DataFrame with edf, ref.df, F, p), `r_squared_adj`, `deviance_explained`, `gcv_score`, `scale_estimate`.
- **`check_diagnostics()`** calls `mgcv::gam.check()` and renders four diagnostic plots via matplotlib: QQ, residuals vs fitted, histogram of residuals, response vs fitted. Also reports basis dimension adequacy (k-index, p-value).
- **`anova()`** with no argument returns an ANOVA table for the model's terms. With another `GAMModel`, performs a likelihood ratio test between nested models.
- **`predict()`** returns a DataFrame with columns: `predicted`, `se` (if `se_fit=True`), `ci_lo`, `ci_hi` (95% CI).

---

## Framework 2: MutationImputer

**Purpose:** Predict binary gene mutation status for subjects lacking genomic panels, using imaging and clinical features as predictors. Trains on the UCSF500 cohort (n=64) that has both genomic and imaging data.

**Statistical approach:** One binomial GAM (`mgcv::gam`, `family="binomial"`) per gene. GAMs chosen over simpler classifiers because they provide proper probability estimates with uncertainty, allow nonlinear imaging-mutation relationships, and share the same `mgcv` backend as `GAMModel`.

### API

```python
class MutationImputer(RModelBase):
    def fit(
        self,
        df: pd.DataFrame,
        gene_columns: list[str],
        predictors: list[str],
        smooth_terms: list[str] = None,
        smooth_k: int | dict = None,
        cv_folds: int = 5,
    ) -> "MutationImputer": ...

    def predict(
        self,
        new_data: pd.DataFrame,
        threshold: float | str = "youden",
    ) -> pd.DataFrame: ...

    def summary(self) -> pd.DataFrame: ...
    def plot_cv_performance(self, genes: list[str] = None): ...
    def plot_feature_importance(self, gene: str): ...
```

### Key Behaviors

- **One model per gene:** Fits a separate binomial GAM for each column in `gene_columns`. Each column should be binary (1=mutated, 0=wild-type).
- **Conservative smooth basis:** Default `smooth_k` should be 5-6 (not `mgcv`'s default of 10) given the small training set (n=64). Overfitting is the primary risk.
- **Built-in cross-validation:** CV is mandatory, not optional. With n=64, honest performance estimates are required before trusting imputation downstream. Stores per-gene AUC, Brier score, and calibration metrics.
- **Threshold selection:** `"youden"` (maximize sensitivity + specificity - 1) is the default. `"prevalence"` matches the observed mutation rate. A float sets a fixed threshold.
- **`predict()`** returns a DataFrame with `{gene}_prob` (predicted probability) and `{gene}_imputed` (binary call) columns for each gene.
- **`summary()`** returns a per-gene table: gene, n_mutated, n_total, prevalence, cv_auc, cv_auc_ci_lo, cv_auc_ci_hi, cv_brier_score, selected_threshold, sensitivity, specificity.
- **`plot_cv_performance()`** shows ROC curves with AUC and calibration plots per gene.
- **`plot_feature_importance()`** shows smooth term partial effects for a specific gene's model — which imaging features drive the prediction.

---

## Framework 3: SurvivalModel

**Purpose:** Comprehensive survival analysis suite. Supports Kaplan-Meier estimation, Cox proportional hazards, penalized Cox (elastic net for high-dimensional features), and accelerated failure time models.

**R packages:** `survival` for KM/Cox/AFT, `glmnet` for penalized Cox.

### API

```python
class SurvivalModel(RModelBase):
    # --- Kaplan-Meier & Log-Rank ---
    def fit_km(self, df, time_col, event_col, group_col=None) -> "SurvivalModel": ...
    def plot_km(self, ci=True, at_risk_table=True, median_line=True, **kwargs): ...
    def logrank_test(self) -> dict: ...

    # --- Cox Proportional Hazards ---
    def fit_cox(self, df, time_col, event_col, predictors, strata=None) -> "SurvivalModel": ...
    def cox_summary(self) -> pd.DataFrame: ...
    def check_proportional_hazards(self) -> pd.DataFrame: ...
    def plot_forest(self, **kwargs): ...

    # --- Penalized Cox (Elastic Net) ---
    def fit_penalized_cox(self, df, time_col, event_col, predictors,
                          alpha=0.5, n_lambda=100, cv_folds=10) -> "SurvivalModel": ...
    def penalized_cox_summary(self, lambda_choice="1se") -> pd.DataFrame: ...
    def plot_regularization_path(self): ...

    # --- Accelerated Failure Time ---
    def fit_aft(self, df, time_col, event_col, predictors,
                distribution="weibull") -> "SurvivalModel": ...
    def aft_summary(self) -> pd.DataFrame: ...
    def compare_aft_distributions(self, df, time_col, event_col,
                                  predictors) -> pd.DataFrame: ...
```

### Key Behaviors

**Kaplan-Meier:**
- Uses `survival::survfit` and `survival::survdiff`.
- `plot_km()` renders publication-ready KM curves via matplotlib with confidence bands, number-at-risk table below the x-axis, and median survival line. Grouped curves get distinct colors and a legend.
- `logrank_test()` returns `chi_sq`, `df`, `p_value`, `n_per_group`, `events_per_group`.

**Cox PH:**
- Uses `survival::coxph`.
- `cox_summary()` returns per-predictor: `coef`, `exp_coef` (hazard ratio), `se`, `z`, `p_value`, `hr_ci_lo`, `hr_ci_hi`, plus model-level `concordance` and `concordance_se`.
- `check_proportional_hazards()` runs `survival::cox.zph` (Schoenfeld residual test). Returns per-predictor `rho`, `chi_sq`, `p_value`. Plots scaled Schoenfeld residuals vs time. This is prominently featured because PH violations are common and the AFT fallback exists for when Cox fails.
- `strata` argument allows stratification on non-PH covariates without estimating their coefficients.
- `plot_forest()` renders a forest plot of hazard ratios with 95% CIs.

**Penalized Cox:**
- Uses `glmnet::cv.glmnet` with `family="cox"`.
- Defaults: `alpha=0.5` (elastic net), `lambda_choice="1se"` (conservative) — appropriate for 150 subjects with 238 imaging features.
- `penalized_cox_summary()` returns non-zero coefficients at the chosen lambda, sorted by absolute value: `predictor`, `coef`, `exp_coef` (HR).
- `plot_regularization_path()` shows coefficient paths vs log(lambda) with the CV error curve and lambda.min/lambda.1se markers.

**AFT Models:**
- Uses `survival::survreg`.
- Supported distributions: `weibull`, `lognormal`, `loglogistic`.
- `aft_summary()` returns per-predictor: `coef`, `se`, `z`, `p_value`, `time_ratio`, `tr_ci_lo`, `tr_ci_hi`, plus `scale` and `distribution`.
- `compare_aft_distributions()` fits all three distributions and returns an AIC/BIC comparison table to guide distribution choice.

**Separate fit methods** (`fit_km`, `fit_cox`, `fit_penalized_cox`, `fit_aft`) rather than one polymorphic `fit()` — these are genuinely different analyses a user would choose deliberately.

---

## Dependencies

**Python (add to `pyproject.toml`):**
```toml
"rpy2>=3.5",
```

**R packages (installed at runtime by `_ensure_r_packages()`):**
- `mgcv` — GAMs (typically pre-installed with R)
- `survival` — Cox PH, KM, AFT (typically pre-installed with R)
- `glmnet` — penalized Cox via elastic net

**Lazy initialization:** `import pcnsl_analysis` never touches R. The R bridge activates only on class instantiation. The rest of the package works fine without R installed.

## Module Exports

Update `__init__.py` to add:

```python
from .pcnsl_analysis import GAMModel, MutationImputer, SurvivalModel
```

## Plotting

All plots render via matplotlib (not R graphics devices) so they work natively in Jupyter notebooks and can be saved with `plt.savefig()`. The R models provide the data; Python handles rendering.

## Error Handling

- Missing R or `rpy2`: clear `ImportError` at instantiation time with install instructions.
- Missing R packages: automatic install prompt via `utils.install.packages()`.
- Unfitted model access (calling `summary()` before `fit()`): `RuntimeError("Model not fitted. Call fit() first.")`.
- Column not found in DataFrame: `ValueError` with the missing column name and available columns.
