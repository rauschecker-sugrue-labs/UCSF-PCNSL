# pcnsl_analysis.py — Design Specification

## Overview

A new module `pcnsl_analysis.py` providing three opinionated analysis frameworks for the UCSF-PCNSL dataset. All backends are pure Python: `pyGAM` for generalized additive models, `lifelines` for survival analysis.

The module is **loosely coupled** to `pcnsl_data_loader.py` — all functions accept pandas DataFrames and column name arguments. No internal data loading.

## Architecture

### No R Dependency

Unlike earlier designs, this module is entirely Python-native. There is no R bridge, no `rpy2`, and no R installation required. `pyGAM` provides GAM fitting via penalized iteratively reweighted least squares, and `lifelines` provides a complete survival analysis toolkit including Kaplan-Meier, Cox PH (with elastic net penalization), and AFT models.

### Shared Base Class

A lightweight `ModelBase` base class (not abstract — no `@abstractmethod` decorators) provides common patterns. `SurvivalModel` uses domain-specific fit methods (`fit_km`, `fit_cox`, etc.) instead of a single `fit()`, so enforcing abstract methods would prevent instantiation.

- **`__init__()`** — no special setup beyond storing configuration.
- **`fit()`** — default raises `NotImplementedError`. Overridden by `GAMModel` and `MutationImputer`. `SurvivalModel` does not override it (uses specific fit methods instead).
- **`predict(new_data)`** — default raises `NotImplementedError`. Overridden by subclasses as appropriate.
- **`summary()`** — default raises `NotImplementedError`. Overridden by subclasses.
- **`self.model_`** — stores the fitted model object after `fit()`.
- **`self.results_`** — stores a Python-side summary dict/DataFrame after `fit()`.

---

## Framework 1: GAMModel

**Purpose:** General-purpose generalized additive model wrapper for predicting any continuous (or other family) response variable. Uses `pyGAM`.

### API

```python
class GAMModel(ModelBase):
    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        predictors: list[str],
        smooth_terms: list[str] = None,
        n_splines: int | dict = None,
        gam_type: str = "linear",
        lam: float | list[float] = None,
    ) -> "GAMModel": ...

    def predict(
        self,
        new_data: pd.DataFrame = None,
        confidence_intervals: bool = True,
        width: float = 0.95,
    ) -> pd.DataFrame: ...

    def summary(self) -> dict: ...
    def plot_partial_dependence(self, terms: list[str] = None, **kwargs): ...
    def check_diagnostics(self): ...
    def gridsearch(self, df: pd.DataFrame = None, target: str = None,
                   lam_candidates: np.ndarray = None) -> "GAMModel": ...
```

### Key Behaviors

- **Term construction:** `predictors` are added as linear terms (`l()`); `smooth_terms` are added as spline terms (`s()`). The user explicitly decides which covariates are nonlinear. Factor/categorical terms use `f()`.
- **`gam_type`** selects the pyGAM class: `"linear"` → `LinearGAM` (identity link, normal distribution), `"logistic"` → `LogisticGAM` (logit link, binomial), `"poisson"` → `PoissonGAM`, `"gamma"` → `GammaGAM`.
- **`n_splines`** controls spline basis dimension per smooth term. Can be an int (applied to all) or a dict mapping column names to individual values. pyGAM default is 20; for small datasets consider reducing to 10-15.
- **`lam`** controls smoothing penalty strength. Can be a single float (applied to all terms) or a list (one per term). If `None`, the user should call `gridsearch()` to select lambda automatically.
- **`gridsearch()`** performs automatic lambda selection via generalized cross-validation. If `df` and `target` are provided, uses them; otherwise replays the training data stored during `fit()` (so `fit()` must be called first, or arguments must be supplied). `lam_candidates` defaults to `np.logspace(-3, 3, 11)`. Internally calls `pyGAM`'s `gridsearch(X, y, lam=...)`. Mutates `self` with the best-fit model and returns `self` for chaining.
- **`summary()`** returns a dict with keys: `explained_deviance` (from `statistics_['pseudo_r2']['explained_deviance']`), `GCV`, `AIC`, `AICc`, `n_samples`, `edof` (total effective degrees of freedom, scalar), `edof_per_term` (list, computed by summing `edof_per_coef` over coefficients belonging to each term), `p_values` (per-term approximate p-values — **caveat:** pyGAM's p-values have a known bug making them unreliable; see pyGAM GitHub issue #163. Do not use for formal inference).
- **`check_diagnostics()`** renders diagnostic plots via matplotlib: QQ plot of deviance residuals, residuals vs fitted, histogram of residuals, response vs fitted.
- **`plot_partial_dependence()`** plots partial dependence (smooth functions) for each term with confidence intervals. Uses `pyGAM`'s `generate_X_grid()` and `partial_dependence()` / `confidence_intervals()`.
- **`predict()`** returns a DataFrame with columns: `predicted`, `ci_lo`, `ci_hi` (if `confidence_intervals=True`). For `LinearGAM`, uses `prediction_intervals()`. For all other GAM types (`LogisticGAM`, `PoissonGAM`, `GammaGAM`), uses `confidence_intervals()` instead (prediction intervals are only available on `LinearGAM`). The intervals are on the response scale in both cases.

---

## Framework 2: MutationImputer

**Purpose:** Predict binary gene mutation status for subjects lacking genomic panels, using imaging and clinical features as predictors. Trains on the UCSF500 cohort (n=64) that has both genomic and imaging data.

**Statistical approach:** One `LogisticGAM` per gene. GAMs chosen over simpler classifiers because they provide proper probability estimates with uncertainty, allow nonlinear imaging-mutation relationships, and pyGAM's `LogisticGAM` handles this natively.

### API

```python
class MutationImputer(ModelBase):
    def fit(
        self,
        df: pd.DataFrame,
        gene_columns: list[str],
        predictors: list[str],
        smooth_terms: list[str] = None,
        n_splines: int | dict = None,
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

- **One model per gene:** Fits a separate `LogisticGAM` for each column in `gene_columns`. Each column should be binary (1=mutated, 0=wild-type).
- **Conservative spline basis:** Default `n_splines` should be 10 (below pyGAM's default of 20) given the small training set (n=64). Overfitting is the primary risk.
- **Minimum mutation count:** Genes with fewer than 5 mutations in the training data are skipped with a warning (too few positive cases for meaningful modeling). These genes appear in `summary()` with `cv_auc=NaN` and a note.
- **Built-in cross-validation:** CV is mandatory, not optional. With n=64, honest performance estimates are required before trusting imputation downstream. Uses **stratified** folds (via `sklearn.model_selection.StratifiedKFold`) to preserve the positive-class ratio in each fold (critical for imbalanced genes). Stores per-gene AUC, Brier score, and calibration metrics. If a per-gene GAM fails to converge during CV, that fold is skipped with a warning; if all folds fail, the gene is skipped entirely.
- **Lambda selection:** Each per-gene `LogisticGAM` uses `gridsearch()` for automatic lambda selection during fitting.
- **Threshold selection:** `"youden"` (maximize sensitivity + specificity - 1) is the default. `"prevalence"` sets the threshold equal to the observed mutation rate in the training data. A float sets a fixed threshold directly.
- **`predict()`** returns a DataFrame with `{gene}_prob` (predicted probability via `LogisticGAM.predict_proba()`, not `predict()` which returns class labels) and `{gene}_imputed` (binary call after thresholding) columns for each gene. Confidence intervals on probabilities use `LogisticGAM.confidence_intervals()` (not `prediction_intervals()`, which is only available on `LinearGAM`).
- **`summary()`** returns a per-gene table: gene, n_mutated, n_total, prevalence, cv_auc, cv_auc_ci_lo, cv_auc_ci_hi (bootstrap CI, not normal approximation — important for small n), cv_brier_score, selected_threshold, sensitivity, specificity.
- **`plot_cv_performance()`** shows ROC curves with AUC and calibration plots per gene.
- **`plot_feature_importance()`** shows partial dependence plots for each term in a specific gene's model — which imaging features drive the prediction.

---

## Framework 3: SurvivalModel

**Purpose:** Comprehensive survival analysis suite. Supports Kaplan-Meier estimation, Cox proportional hazards (including penalized/elastic net), and accelerated failure time models. All via `lifelines`.

### API

```python
class SurvivalModel(ModelBase):
    # --- Kaplan-Meier & Log-Rank ---
    def fit_km(self, df, time_col, event_col, group_col=None) -> "SurvivalModel": ...
    def plot_km(self, ci=True, at_risk_table=True, median_line=True, **kwargs): ...
    def logrank_test(self) -> dict: ...

    # --- Cox Proportional Hazards ---
    def fit_cox(self, df, time_col, event_col, predictors, strata=None,
                penalizer=0.0, l1_ratio=0.0) -> "SurvivalModel": ...
    def cox_summary(self) -> pd.DataFrame: ...
    def check_proportional_hazards(self) -> pd.DataFrame: ...
    def plot_forest(self, **kwargs): ...

    # --- Penalized Cox (Elastic Net) ---
    def fit_penalized_cox(self, df, time_col, event_col, predictors,
                          penalizer=0.1, l1_ratio=0.5) -> "SurvivalModel": ...
    def penalized_cox_summary(self) -> pd.DataFrame: ...

    # --- Predictions for new data ---
    def predict_survival_function(self, new_data: pd.DataFrame) -> pd.DataFrame: ...
    def predict_median(self, new_data: pd.DataFrame) -> pd.Series: ...
    def predict_hazard(self, new_data: pd.DataFrame) -> pd.DataFrame: ...

    # --- Accelerated Failure Time ---
    def fit_aft(self, df, time_col, event_col, predictors,
                distribution="weibull") -> "SurvivalModel": ...
    def aft_summary(self) -> pd.DataFrame: ...
    def compare_aft_distributions(self, df, time_col, event_col,
                                  predictors) -> pd.DataFrame: ...
```

### Key Behaviors

**Kaplan-Meier:**
- Uses `lifelines.KaplanMeierFitter`.
- `fit_km()` with no `group_col` fits a single curve. With `group_col`, fits one `KaplanMeierFitter` per group and stores them in `self.km_models_` (dict keyed by group label).
- `plot_km()` renders publication-ready KM curves via `KaplanMeierFitter.plot_survival_function()` with confidence bands. When `at_risk_table=True`, uses `lifelines.plotting.add_at_risk_counts()`. When `median_line=True`, draws a horizontal line at 0.5 intersecting each curve's median. Grouped curves get distinct colors and a legend.
- `logrank_test()` uses `lifelines.statistics.logrank_test`. Returns `test_statistic`, `p_value`, `n_per_group`, `events_per_group`. Raises `RuntimeError` if `group_col` was not used in `fit_km()`.

**Cox PH:**
- Uses `lifelines.CoxPHFitter`.
- `fit_cox()` accepts `penalizer` (penalty strength, default 0.0 = no penalty) and `l1_ratio` (mix between L1 and L2: 0.0 = pure L2, 1.0 = pure L1, default 0.0). The penalty formula is `penalizer * ((1 - l1_ratio)/2 * ||beta||_2^2 + l1_ratio * ||beta||_1)`. The DataFrame passed to `CoxPHFitter.fit()` includes `[time_col, event_col] + predictors + (strata or [])` — strata columns must be in the DataFrame. `strata` is also passed to the `strata` parameter of `.fit()`.
- `cox_summary()` returns a tuple `(summary_df, metadata_dict)`. `summary_df` is `CoxPHFitter.summary` — columns: `coef`, `exp(coef)` (hazard ratio), `se(coef)`, `coef lower 95%`, `coef upper 95%`, `exp(coef) lower 95%`, `exp(coef) upper 95%`, `z`, `p`, `-log2(p)`. `metadata_dict` contains `concordance_index` (`CoxPHFitter.concordance_index_`) and `log_likelihood` (`CoxPHFitter.log_likelihood_`).
- `check_proportional_hazards()` calls `CoxPHFitter.check_assumptions()`, which prints Schoenfeld residual test results to stdout and produces diagnostic plots. This method captures the printed output, parses per-predictor test statistics and p-values into a DataFrame (`predictor`, `test_statistic`, `p_value`), and returns it. The Schoenfeld residual plots are rendered to the current matplotlib figure. If parsing fails, falls back to returning `None` and printing a message directing the user to the console output.
- `strata` argument allows stratification on non-PH covariates without estimating their coefficients.
- `plot_forest()` renders a forest plot of hazard ratios with 95% CIs. Uses custom matplotlib rendering from the summary DataFrame (horizontal error bars at each predictor). `lifelines`' `CoxPHFitter.plot()` can also produce this with `hazard_ratios=True`, but custom rendering gives more control for publication quality.

**Penalized Cox:**
- Also uses `lifelines.CoxPHFitter` but with `penalizer > 0` and `l1_ratio` between 0 and 1 for elastic net behavior.
- `fit_penalized_cox()` is a convenience wrapper around `fit_cox()` with stronger defaults: `penalizer=0.1`, `l1_ratio=0.5` (elastic net). Appropriate for 150 subjects with 238 imaging features.
- `penalized_cox_summary()` returns non-zero coefficients (those surviving penalization) sorted by absolute value: `predictor`, `coef`, `exp(coef)` (HR), `se(coef)`, `p`.
- Note: `lifelines` does not provide a built-in regularization path plot or cross-validated lambda selection like `glmnet`. If cross-validated penalizer selection is needed, the user should use `sklearn.model_selection.GridSearchCV` with lifelines' concordance index as the scoring metric, or manually loop over penalizer values. A helper method `select_penalizer()` can be added to automate this pattern.

**Predictions for new data:**
- `predict_survival_function()`, `predict_median()`, and `predict_hazard()` delegate to the most recently fitted Cox or AFT model's corresponding lifelines methods (`CoxPHFitter.predict_survival_function()`, etc.). Raises `RuntimeError` if no Cox or AFT model has been fitted.
- These are essential for the penalized Cox workflow where users select features from 238 imaging columns and want to predict on held-out data.

**AFT Models:**
- Uses `lifelines.WeibullAFTFitter`, `lifelines.LogNormalAFTFitter`, `lifelines.LogLogisticAFTFitter`.
- `distribution` selects the fitter class: `"weibull"`, `"lognormal"`, `"loglogistic"`.
- `aft_summary()` returns the fitter's `.summary` DataFrame with per-predictor: `coef`, `se(coef)`, `z`, `p`, `coef lower 95%`, `coef upper 95%`, plus `AIC_` from the model.
- `compare_aft_distributions()` fits all three distributions and returns an AIC/BIC comparison table to guide distribution choice. Uses each fitter's `AIC_` and `BIC_` properties directly (available on all lifelines AFT fitters).

**Separate fit methods** (`fit_km`, `fit_cox`, `fit_penalized_cox`, `fit_aft`) rather than one polymorphic `fit()` — these are genuinely different analyses a user would choose deliberately. All return `-> "SurvivalModel"` for chaining.

**Multi-model state management:** Each fit method stores its result under a distinct attribute: `self.km_models_` (dict or single fitter), `self.cox_model_`, `self.penalized_cox_model_`, `self.aft_model_`. A single `SurvivalModel` instance can hold all four simultaneously. Summary/plot methods raise `RuntimeError("No KM model fitted. Call fit_km() first.")` (etc.) if the corresponding model attribute is `None`.

**`compare_aft_distributions()`** is self-contained — it accepts all arguments and does not depend on prior `fit_aft()` state. This is intentional: it fits three models internally for comparison and does not mutate `self.aft_model_`.

---

## Dependencies

**Python (add to `pyproject.toml` as an optional dependency):**
```toml
[project.optional-dependencies]
analysis = ["lifelines>=0.29", "pygam>=0.9", "scikit-learn>=1.3"]
```

- `lifelines` — Kaplan-Meier, Cox PH (with penalization), AFT models
- `pygam` — generalized additive models (LinearGAM, LogisticGAM, etc.)
- `scikit-learn` — `StratifiedKFold` for cross-validation in `MutationImputer`, ROC/AUC metrics

**Lazy initialization:** `import pcnsl_analysis` should not fail if these optional dependencies are missing. Classes raise `ImportError` with install instructions on instantiation if the required packages are not available.

## Module Exports

Update `__init__.py` with a guarded import so users without the analysis dependencies can still use the rest of the package:

```python
try:
    from .pcnsl_analysis import GAMModel, MutationImputer, SurvivalModel
except ImportError:
    pass
```

Users without `lifelines`/`pygam` can still `from pcnsl_data_loader import PCNSLDataLoader`. Users who need analysis import directly: `from pcnsl_analysis import GAMModel`.

## Plotting

All plots render via matplotlib/seaborn so they work natively in Jupyter notebooks and can be saved with `plt.savefig()`. `lifelines` plotting functions integrate with matplotlib axes natively. `pyGAM` partial dependence data is extracted and plotted via matplotlib.

## Missing Data (NA) Handling

All three frameworks use a consistent NA strategy: **drop rows with any NA in the target or predictor columns before fitting, and emit a warning stating how many rows were dropped.** The original DataFrame is never mutated — a filtered copy is used internally.

For `predict()`, rows with NA in predictor columns get `NaN` in the output (no prediction) with a warning.

## Error Handling

- Missing `lifelines` or `pygam`: clear `ImportError` at instantiation time with install instructions (e.g., `pip install lifelines pygam` or `uv add lifelines pygam`).
- Unfitted model access (calling `summary()` before `fit()`): `RuntimeError("Model not fitted. Call fit() first.")`.
- Column not found in DataFrame: `ValueError` with the missing column name and available columns.
