import marimo

__generated_with = "0.9.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""# Applying Split Conformal Prediction to a Motor Insurance Regression Problem"""
    )
    return


@app.cell
def __():
    import os
    import warnings
    from math import inf
    from pathlib import Path
    from typing import Callable, Literal

    import altair as alt
    import marimo as mo
    import numpy as np
    import optuna
    import pandas as pd
    import requests
    from requests.exceptions import RequestException
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import median_absolute_error, root_mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        KBinsDiscretizer,
        OneHotEncoder,
        OrdinalEncoder,
        StandardScaler,
    )
    from sklearn.utils.validation import check_is_fitted

    return (
        BaseEstimator,
        Callable,
        ColumnTransformer,
        HistGradientBoostingRegressor,
        KBinsDiscretizer,
        Literal,
        OneHotEncoder,
        OrdinalEncoder,
        Path,
        Pipeline,
        RegressorMixin,
        RequestException,
        StandardScaler,
        alt,
        check_is_fitted,
        inf,
        median_absolute_error,
        mo,
        np,
        optuna,
        os,
        pd,
        requests,
        root_mean_squared_error,
        train_test_split,
        warnings,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Loading and Inspecting the `freMTPL2` Dataset""")
    return


@app.cell(hide_code=True)
def __(Path, mo, np, pd, requests, warnings):
    def load_fremtpl2_dataset(
        save_dir: Path | str | None = None, verbose: bool = False
    ) -> pd.DataFrame:

        csv_paths = download_fremtpl2_dataset(save_dir=save_dir)

        df_freq = pd.read_csv(
            csv_paths["freq"],
            dtype={
                "IDpol": "int64",  # Policy ID
                "ClaimNb": "int64",  # Number of claims
                "Exposure": "float64",  # Exposure period in years
                "Area": "category",  # Ordinal encoding of population density
                "VehPower": "int64",  # Vehicle power category
                "VehAge": "int64",  # Vehicle age 'last birthday' in years
                "DrivAge": "int64",  # Driver age 'last birthday' in years
                "BonusMalus": "int64",  # Bonus-malus factor
                "VehBrand": "category",  # Vehicle brand
                "VehGas": "category",  # Vehicle fuel type
                "Density": "float64",  # Population density
                "Region": "category",  # Region in France where policyholder resides
            },
        )

        df_sev = pd.read_csv(
            csv_paths["sev"],
            dtype={
                "IDpol": "int64",  # Policy ID
                "ClaimAmount": "float64",  # Loss amount in an unspecified currency (presumably Euros)
            },
        )

        # Create list of `ClaimAmount`s for each policy:
        df_sev = df_sev.groupby("IDpol")["ClaimAmount"].apply(list).reset_index()
        df_sev = df_sev.rename(columns={"ClaimAmount": "Claims"})

        # Join entire `ClaimAmount` history of each policy to exposure data:
        df = df_freq.merge(
            df_sev, how="outer", on="IDpol", validate="one_to_one", indicator=True
        )
        unmatched_claims = df["_merge"] == "right_only"
        if verbose and np.any(unmatched_claims):
            num_unmtch_policies = int(unmatched_claims.sum())
            num_unmtch_claims = int(
                df.loc[unmatched_claims, "Claims"].apply(lambda x: len(x)).sum()
            )
            warnings.warn(
                f"{num_unmtch_policies:,} policies referenced in `freMTPL2sev.csv` do not have "
                f"matching records in `freMTPL2freq.csv`. The {num_unmtch_claims:,} claims "
                "linked to these policies will be excluded from the analysis."
            )
        df = df.drop(columns="_merge")
        df = df[~unmatched_claims].reset_index(drop=True)

        df["NumClaims"] = df["Claims"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        df["TotalLoss"] = df["Claims"].apply(
            lambda x: sum(x) if isinstance(x, list) else 0
        )
        incorr_claim_count = df["NumClaims"] != df["ClaimNb"]
        if verbose and np.any(incorr_claim_count):
            num_incorr_count = int(incorr_claim_count.sum())
            warnings.warn(
                f"{num_incorr_count:,} policies in `freMTPL2freq.csv` have a `ClaimNb` value that "
                f"does not match the actual number of claims records in `freMTPL2sev.csv`. "
                "The `ClaimNb` for these policies will be set to the number of claim records "
                "associated with those policies in `freMTPL2sev.csv`."
            )
        df["ClaimNb"] = df["NumClaims"]
        df = df.drop(columns="NumClaims")

        return df

    def download_fremtpl2_dataset(
        save_dir: Path | str | None = None, chunk_size: int = 8192
    ) -> dict[str, Path]:

        url_fstr = "https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main/{filename}?download=true"

        if save_dir is None:
            save_dir = Path(".")
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        dataset_paths: dict[str, Path] = {}
        for filename in ("freMTPL2freq.csv", "freMTPL2sev.csv"):
            dataset_path = save_dir / Path(filename)
            if not dataset_path.exists():
                dataset_url = url_fstr.format(filename=str(filename))
                try:
                    with requests.get(dataset_url, stream=True) as response:
                        response.raise_for_status()
                        total_size = int(response.headers.get("Content-length"))
                        with mo.status.progress_bar(
                            total=total_size, title=f"Downloading `{str(filename)}`..."
                        ) as pbar:
                            with open(dataset_path, "wb") as file:
                                for chunk in response.iter_content(
                                    chunk_size=chunk_size
                                ):
                                    file.write(chunk)
                                    pbar.update(len(chunk))
                except requests.RequestException as e:
                    raise requests.RequestException(
                        f"Failed to download `{str(filename)}` from `huggingface.co`. "
                        "If you're behind a corporate firewall, please manually download this "
                        f"dataset from {dataset_url} and save it as '{str(filename)}' in the "
                        f"directory '{str(save_dir.absolute())}'."
                    ) from e
            freq_or_sev = "freq" if str(filename).endswith("freq.csv") else "sev"
            dataset_paths[freq_or_sev] = dataset_path

        return dataset_paths

    return download_fremtpl2_dataset, load_fremtpl2_dataset


@app.cell
def __(load_fremtpl2_dataset):
    data = load_fremtpl2_dataset()
    data["Frequency"] = data["ClaimNb"] / data["Exposure"]
    return (data,)


@app.cell
def __(data):
    data
    return


@app.cell
def __(pd):
    def explode_claims(df: pd.DataFrame) -> pd.DataFrame:
        claims = df.explode("Claims", ignore_index=True)
        claims = claims.dropna(subset=["Claims"])
        claims = claims.rename(columns={"Claims": "ClaimAmount"})
        claims = claims.reset_index(drop=True)
        return claims

    return (explode_claims,)


@app.cell
def __(data, explode_claims):
    explode_claims(data)
    return


@app.cell(hide_code=True)
def __(alt, explode_claims, np, pd):
    def plot_fremtpl2_freq_dist(data: pd.DataFrame):
        freq, freq_counts = np.unique(data["ClaimNb"], return_counts=True)
        freq_counts = np.where(freq_counts == 1, 1.25, freq_counts)

        df_freq = pd.DataFrame({"freq": freq, "log10_count": np.log10(freq_counts)})
        freq_plot = (
            alt.Chart(df_freq)
            .mark_bar()
            .encode(
                x=alt.X(
                    "freq:Q",
                    title="Number of Claims against Policy",
                    scale=alt.Scale(domain=[freq.min(), freq.max()]),
                ),
                y=alt.Y("log10_count:Q", title="log₁₀(Number of Policies)"),
            )
            .properties(title="Frequency Distribution")
        )

        return freq_plot

    def plot_fremtpl2_sev_data(
        data: pd.DataFrame, epsilon: float = 1e-5, nbins: int = 30
    ) -> alt.Chart:
        claim_amts = explode_claims(data)["ClaimAmount"].to_numpy(dtype=float)
        hist_values, bin_edges = np.histogram(
            np.log10(claim_amts + epsilon), bins=nbins
        )
        with np.errstate(divide="ignore"):
            log10_hist_values = np.log10(hist_values)

        df_sev = pd.DataFrame(
            {
                "bin_start": bin_edges[1:],
                "bin_end": bin_edges[:-1],
                "count": log10_hist_values,
            }
        )
        sev_plot = (
            alt.Chart(df_sev)
            .mark_bar()
            .encode(
                x=alt.X(
                    "bin_start:Q",
                    title="log₁₀(Claim Amount)",
                    bin="binned",
                ),
                x2=alt.X2("bin_end:Q"),
                y=alt.Y("count:Q", title="log₁₀(Number of Claims)"),
            )
            .properties(title="Severity Distribution")
        )

        return sev_plot

    return plot_fremtpl2_freq_dist, plot_fremtpl2_sev_data


@app.cell(hide_code=True)
def __(data, plot_fremtpl2_freq_dist, plot_fremtpl2_sev_data):
    plot_fremtpl2_freq_dist(data).properties(
        width=400, height=300
    ) | plot_fremtpl2_sev_data(data).properties(width=400, height=300)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Data Splits""")
    return


@app.cell
def __(np, pd, train_test_split):
    def train_test_calib_valid_split(
        df: pd.DataFrame,
        train_frac: float,
        test_frac: float,
        calib_frac: float,
        rng: np.random.Generator,
        stratify: str,
        nbins: int = 10,
    ) -> tuple[pd.DataFrame, ...]:

        try:
            assert train_frac + test_frac + calib_frac < 1
        except AssertionError:
            total = train_frac + test_frac + calib_frac
            raise ValueError(
                "Sum of `train_frac`, `test_frac`, and `calib_frac` "
                f"(= {total:.2f}) must be less than 1"
            )

        quantiles = qcut_with_tie_breaking(
            df[stratify].to_numpy(), nbins=nbins, rng=rng
        )

        train_size = np.round(train_frac * len(df)).astype(int)
        test_size = np.round(test_frac * len(df)).astype(int)
        calib_size = np.round(calib_frac * len(df)).astype(int)

        df_train, df_not_train, _, quantiles_not_train = train_test_split(
            df,
            quantiles,
            train_size=train_size,
            stratify=quantiles,
            random_state=sample_random_seed(rng),
        )
        df_test, df_not_test, _, quantiles_not_test = train_test_split(
            df_not_train,
            quantiles_not_train,
            train_size=test_size,
            stratify=quantiles_not_train,
            random_state=sample_random_seed(rng),
        )
        df_calib, df_valid = train_test_split(
            df_not_test,
            train_size=calib_size,
            stratify=quantiles_not_test,
            random_state=sample_random_seed(rng),
        )

        return df_train, df_test, df_calib, df_valid

    def qcut_with_tie_breaking(
        x: np.ndarray, nbins: int, rng: np.random.Generator, epsilon: float = 1e-5
    ) -> np.ndarray:
        tie_breaking_noise = sample_tie_breaking_noise(
            size=x.size, rng=rng, epsilon=epsilon
        )
        return pd.qcut(x + tie_breaking_noise, q=nbins, labels=False)

    def sample_tie_breaking_noise(
        size: int, rng: np.random.Generator, epsilon: float = 1e-5
    ):
        return epsilon * rng.uniform(low=0, high=1, size=size)

    def sample_random_seed(rng: np.random.Generator) -> int:
        return rng.integers(low=0, high=np.iinfo(np.int32).max)

    return (
        qcut_with_tie_breaking,
        sample_random_seed,
        sample_tie_breaking_noise,
        train_test_calib_valid_split,
    )


@app.cell
def __():
    train_frac: float = 0.3
    test_frac: float = 0.1
    calib_frac: float = 0.2
    seed_sample: int = 42
    return calib_frac, seed_sample, test_frac, train_frac


@app.cell
def __(
    calib_frac,
    data,
    np,
    seed_sample,
    test_frac,
    train_frac,
    train_test_calib_valid_split,
):
    df_train, df_test, df_calib, df_valid = train_test_calib_valid_split(
        df=data,
        train_frac=train_frac,
        test_frac=test_frac,
        calib_frac=calib_frac,
        rng=np.random.default_rng(seed_sample),
        stratify="TotalLoss",
    )
    return df_calib, df_test, df_train, df_valid


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Training Simple GBM-Based Frequency and Severity Models""")
    return


@app.cell
def __(
    ColumnTransformer,
    HistGradientBoostingRegressor,
    Literal,
    OneHotEncoder,
    OrdinalEncoder,
    Pipeline,
    StandardScaler,
    np,
    pd,
    sample_random_seed,
):
    def train_gbm_model(
        data: pd.DataFrame,
        model_type: Literal["freq", "sev"],  # noqa: F821
        rng: np.random.Generator,
        **gbm_params,
    ) -> Pipeline:
        feature_preprocessor = ColumnTransformer(
            [
                ("categorical", OneHotEncoder(), ["VehBrand", "VehGas", "Region"]),
                (
                    "numeric",
                    StandardScaler(),
                    ["VehAge", "DrivAge", "BonusMalus", "Density"],
                ),
                ("ordinal", OrdinalEncoder(), ["VehPower", "Area"]),
            ],
            # Only use columns listed above as input features; ignore all others (e.g. IDpol)
            remainder="drop",
            # Ensure only dense arrays are passed to regressor
            sparse_threshold=0,
        )
        feature_preprocessor.fit(data)
        if model_type == "freq":
            objective = "poisson"
        elif model_type == "sev":
            objective = "gamma"
        else:
            raise ValueError("TODO.")

        gbm_model = HistGradientBoostingRegressor(
            loss=objective, random_state=sample_random_seed(rng), **gbm_params
        )

        model = Pipeline([("preprocessor", feature_preprocessor), ("gbm", gbm_model)])

        if model_type == "freq":
            # Fitting Poisson model to frequencies w/ exposures used as sample weights
            # is equivalent to fitting Poisson model to counts w/ a log exposure offset.
            # See https://stats.stackexchange.com/a/270151 for proof.
            model.fit(X=data, y=data["Frequency"], gbm__sample_weight=data["Exposure"])
        elif model_type == "sev":
            model.fit(X=data, y=data["ClaimAmount"])

        return model

    return (train_gbm_model,)


@app.cell
def __(
    Literal,
    np,
    optuna,
    pd,
    root_mean_squared_error,
    sample_random_seed,
    train_gbm_model,
):
    def optimize_gbm_model_hyperparameters(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        model_type: Literal["freq", "sev"],  # noqa: F821
        rng: np.random.Generator,
        ntrials: int = 5,
        verbose: bool = False,
    ):
        """
        Takes heavy inspiration from:
            https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/
        """

        def objective_fn(trial: optuna.Trial) -> float:
            gbm_params = {
                "max_iter": trial.suggest_int("max_iter", 50, 500),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 0.1, log=True
                ),
                "l2_regularization": trial.suggest_float(
                    "l2_regularization", 1e-3, 0.1, log=True
                ),
            }
            model = train_gbm_model(
                data=train_data, model_type=model_type, rng=rng, **gbm_params
            )
            y_pred = model.predict(test_data)
            if model_type == "freq":
                test_loss = root_mean_squared_error(
                    test_data["ClaimNb"], test_data["Exposure"] * y_pred
                )
            elif model_type == "sev":
                test_loss = root_mean_squared_error(test_data["ClaimAmount"], y_pred)
            else:
                raise ValueError("TODO")
            return test_loss

        current_verbosity = optuna.logging.get_verbosity()
        sampler = optuna.samplers.TPESampler(seed=sample_random_seed(rng))
        try:
            optuna.logging.set_verbosity(
                optuna.logging.WARNING if not verbose else current_verbosity
            )
            study = optuna.create_study(sampler=sampler, direction="minimize")
            study.optimize(objective_fn, n_trials=ntrials, show_progress_bar=verbose)
        finally:
            optuna.logging.set_verbosity(current_verbosity)
        return study.best_params

    return (optimize_gbm_model_hyperparameters,)


@app.cell
def __(df_test, df_train, np, optimize_gbm_model_hyperparameters):
    seed_hyperparam_freq = 123
    freq_hyperparams = optimize_gbm_model_hyperparameters(
        train_data=df_train,
        test_data=df_test,
        model_type="freq",
        rng=np.random.default_rng(seed_hyperparam_freq),
        ntrials=5,
        verbose=True,
    )
    return freq_hyperparams, seed_hyperparam_freq


@app.cell
def __(df_calib, df_test, df_train, df_valid, explode_claims):
    df_train_claims = explode_claims(df_train)
    df_test_claims = explode_claims(df_test)
    df_calib_claims = explode_claims(df_calib)
    df_valid_claims = explode_claims(df_valid)
    return df_calib_claims, df_test_claims, df_train_claims, df_valid_claims


@app.cell
def __(
    df_test_claims,
    df_train_claims,
    np,
    optimize_gbm_model_hyperparameters,
):
    seed_hyperparam_sev = 456
    sev_hyperparams = optimize_gbm_model_hyperparameters(
        train_data=df_train_claims,
        test_data=df_test_claims,
        model_type="sev",
        rng=np.random.default_rng(seed_hyperparam_sev),
        ntrials=5,
        verbose=True,
    )
    return seed_hyperparam_sev, sev_hyperparams


@app.cell
def __(df_train, df_train_claims, np, train_gbm_model):
    seed_train = 789
    model_freq = train_gbm_model(
        data=df_train, model_type="freq", rng=np.random.default_rng(seed_train)
    )
    model_sev = train_gbm_model(
        data=df_train_claims,
        model_type="sev",
        rng=np.random.default_rng(seed_train + 1),
    )
    return model_freq, model_sev, seed_train


@app.cell
def __(
    df_train,
    df_train_claims,
    df_valid,
    df_valid_claims,
    model_freq,
    model_sev,
):
    freq_train_pred = model_freq.predict(df_train) * df_train["Exposure"]
    sev_train_pred = model_sev.predict(df_train_claims)
    freq_valid_pred = model_freq.predict(df_valid) * df_valid["Exposure"]
    sev_valid_pred = model_sev.predict(df_valid_claims)
    return freq_train_pred, freq_valid_pred, sev_train_pred, sev_valid_pred


@app.cell(hide_code=True)
def __(Literal, alt, np, pd, qcut_with_tie_breaking):
    def plot_model_residuals(
        y_pred: np.ndarray,
        residuals: np.ndarray,
        model_type: Literal["freq", "sev"],  # noqa: F821
        sampling_frac: float | None = None,
        seed: int | None = None,
        sampling_nquantiles: int = 10,
        marker_size: int = 30,
        marker_opacity: float = 0.5,
        hline_color: str = "orange",
        hline_strokedash: tuple[int, int] = (5, 5),
        xlims: tuple[float, float] | None = None,
        ylims: tuple[float, float] | None = None,
    ) -> alt.Chart:

        alt.data_transformers.enable("vegafusion")

        df = pd.DataFrame({"y_pred": y_pred, "residual": residuals})

        if xlims is not None:
            df = df[(df["y_pred"] >= xlims[0]) & (df["y_pred"] <= xlims[1])]
            xscale = alt.Scale(domain=xlims)
        else:
            xscale = alt.Undefined

        if ylims is not None:
            df = df[(df["residual"] >= ylims[0]) & (df["residual"] <= ylims[1])]
            yscale = alt.Scale(domain=ylims)
        else:
            yscale = alt.Undefined

        if (sampling_frac is not None) and (seed is not None):
            df["res_quantile"] = pd.qcut(df["residual"], q=sampling_nquantiles)
            df_samples = df.groupby("res_quantile", observed=False).sample(
                frac=sampling_frac, replace=False, random_state=seed
            )
        elif (sampling_frac is not None) and (seed is None):
            raise ValueError("TODO")
        else:
            df_samples = df

        if model_type == "freq":
            y_title = "Predicted Claim Count"
        elif model_type == "sev":
            y_title = "Predicted Claim Amount"
        else:
            raise ValueError("TODO")

        scatter_plot = (
            alt.Chart(df_samples)
            .mark_circle(size=marker_size, fillOpacity=marker_opacity)
            .encode(
                x=alt.X("y_pred:Q", title=y_title, scale=xscale),
                y=alt.Y(
                    "residual:Q",
                    title="Residual (= Observed - Prediction)",
                    scale=yscale,
                ),
            )
        )

        df_hline = pd.DataFrame({"y": [0]})
        hline_plot = (
            alt.Chart(df_hline)
            .mark_rule(strokeDash=hline_strokedash, color=hline_color)
            .encode(
                y=alt.Y("y:Q"),
            )
        )

        return scatter_plot + hline_plot

    def plot_total_response_by_pred_quantile(
        y_pred: np.ndarray,
        y_obs: np.ndarray,
        model_type: Literal["freq", "sev"],  # noqa: F821
        nquantiles: int = 10,
        seed: int = 42,
    ) -> alt.Chart:
        y_pred_quantiles = qcut_with_tie_breaking(
            y_pred, nbins=nquantiles, rng=np.random.default_rng(seed)
        )
        df = pd.DataFrame(
            {
                "y_pred": y_pred,
                "y_obs": y_obs,
                "quantile": y_pred_quantiles,
            }
        )

        df_plot = pd.DataFrame(
            {
                "quantile": np.linspace(0, 1, nquantiles),
                "Predicted": df.groupby("quantile")["y_pred"].sum(),
                "Observed": df.groupby("quantile")["y_obs"].sum(),
            }
        )

        df_plot = df_plot.melt(id_vars=["quantile"], var_name="y", value_name="sum")

        if model_type == "freq":
            x_title = "Predicted Claim Count Quantile"
            y_title = "Total Claim Count"
        elif model_type == "sev":
            x_title = "Predicted Claim Amount Quantile"
            y_title = "Total Claim Amount"
        else:
            raise ValueError("TODO")

        plot = (
            alt.Chart(df_plot)
            .mark_line(point=alt.OverlayMarkDef())
            .encode(
                x=alt.X("quantile:Q", title=x_title, axis=alt.Axis(format=".0%")),
                y=alt.Y("sum:Q", title=y_title),
                color=alt.Color("y:N", title="Legend"),
                strokeDash=alt.StrokeDash("y:N", title="Legend"),
            )
        )

        return plot

    return plot_model_residuals, plot_total_response_by_pred_quantile


@app.cell(hide_code=True)
def __(
    df_valid,
    freq_valid_pred,
    plot_model_residuals,
    plot_total_response_by_pred_quantile,
):
    plot_model_residuals(
        y_pred=freq_valid_pred,
        residuals=df_valid["ClaimNb"] - freq_valid_pred,
        model_type="freq",
        sampling_frac=0.1,
        seed=42,
    ).properties(
        title="Frequency Model Residuals on Validation Dataset"
    ) | plot_total_response_by_pred_quantile(
        y_pred=freq_valid_pred, y_obs=df_valid["ClaimNb"], model_type="freq"
    ).properties(
        title="Total Claim Count vs Prediction Quantile on Validation Dataset"
    )
    return


@app.cell(hide_code=True)
def __(df_train, df_valid, freq_train_pred, freq_valid_pred):
    print(
        f"Predicted total number of claims (training data): {freq_train_pred.sum().astype(int):,}"
    )
    print(
        f'Observed total number of claims (training data): {df_train["ClaimNb"].sum():,}'
    )
    print(
        f"Predicted total number of claims (validation data): {freq_valid_pred.sum().astype(int):,}"
    )
    print(
        f'Observed total number of claims (validation data): {df_valid["ClaimNb"].sum():,}'
    )
    return


@app.cell(hide_code=True)
def __(
    df_train_claims,
    plot_model_residuals,
    plot_total_response_by_pred_quantile,
    sev_train_pred,
):
    plot_model_residuals(
        y_pred=sev_train_pred,
        residuals=df_train_claims["ClaimAmount"] - sev_train_pred,
        model_type="sev",
        xlims=(0, 20_000),
    ).properties(
        title="Severity Model Residuals on Training Dataset"
    ) | plot_total_response_by_pred_quantile(
        y_pred=sev_train_pred, y_obs=df_train_claims["ClaimAmount"], model_type="sev"
    ).properties(
        title="Total Claim Amount vs Prediction Quantile on Training Dataset"
    )
    return


@app.cell(hide_code=True)
def __(df_train_claims, df_valid_claims, sev_train_pred, sev_valid_pred):
    print(f"Predicted total loss (training data): {sev_train_pred.sum():,.2f}")
    print(
        f'Observed total loss (training data): {df_train_claims["ClaimAmount"].sum():,.2f}'
    )
    print(f"Predicted total loss (validation data): {sev_valid_pred.sum():,.2f}")
    print(
        f'Observed total loss (validation data): {df_valid_claims["ClaimAmount"].sum():,.2f}'
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Conformal Calibration of Frequency and Severity Models""")
    return


@app.cell
def __(np, sample_tie_breaking_noise):
    def calibrate_regressor(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        alpha: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        calib_scores = np.abs(y_true - y_pred)
        calib_scores = calib_scores + sample_tie_breaking_noise(
            size=calib_scores.size, rng=rng
        )
        sorted_scores = np.sort(calib_scores)
        qhats_idx: np.ndarray = np.ceil((calib_scores.size + 1) * (1 - alpha)) - 1
        qhats_idx = qhats_idx.astype(int)
        qhat = np.append(sorted_scores, np.inf)[qhats_idx]
        return qhat

    return (calibrate_regressor,)


@app.cell
def __(np, qhats):
    def compute_prediction_sets(
        y_pred: np.ndarray,
        qhat: np.ndarray,
        assume_positive: bool,
    ) -> np.ndarray:
        num_alpha = qhats.shape[0]
        num_samples = y_pred.size
        assert qhats.ndim == 1
        assert y_pred.ndim == 1
        ub = y_pred + qhats.reshape(-1, 1)
        lb = y_pred - qhats.reshape(-1, 1)
        pred_sets = np.stack([lb, ub], axis=2)
        pred_sets = pred_sets.astype(float)
        if assume_positive:
            pred_sets = np.clip(pred_sets, a_min=0, a_max=np.inf)
        assert pred_sets.shape == (num_alpha, num_samples, 2)
        return pred_sets

    return (compute_prediction_sets,)


@app.cell
def __(np):
    def compute_coverage(y_valid: np.ndarray, y_lb: np.ndarray, y_ub: np.ndarray):
        return np.mean((y_lb <= y_valid) & (y_valid <= y_ub))

    return (compute_coverage,)


@app.cell
def __():
    alpha = 0.01
    return (alpha,)


@app.cell
def __(alpha, calibrate_regressor, df_calib, model_freq, np):
    seed_calib = 123
    freq_calib_pred = model_freq.predict(df_calib) * df_calib["Exposure"]
    qhat_freq = calibrate_regressor(
        y_pred=freq_calib_pred,
        y_true=df_calib["ClaimNb"],
        alpha=alpha,
        rng=np.random.default_rng(seed_calib),
    )
    return freq_calib_pred, qhat_freq, seed_calib


@app.cell
def __(
    alpha,
    calibrate_regressor,
    df_calib_claims,
    model_sev,
    np,
    seed_calib,
):
    sev_calib_pred = model_sev.predict(df_calib_claims)
    qhat_sev = calibrate_regressor(
        y_pred=sev_calib_pred,
        y_true=df_calib_claims["ClaimAmount"],
        alpha=alpha,
        rng=np.random.default_rng(seed_calib),
    )
    return qhat_sev, sev_calib_pred


@app.cell
def __(freq_valid_pred, np, qhat_freq):
    freq_valid_lb = np.clip(freq_valid_pred - qhat_freq, a_min=0, a_max=np.inf)
    freq_valid_ub = freq_valid_pred + qhat_freq
    return freq_valid_lb, freq_valid_ub


@app.cell(hide_code=True)
def __(Literal, alt, np, pd, qcut_with_tie_breaking):
    def plot_prediction_intervals_with_validation_points(
        y_pred: np.ndarray,
        y_obs: np.ndarray,
        y_lb: np.ndarray,
        y_ub: np.ndarray,
        model_type: Literal["freq", "sev", "loss"],  # noqa: F821
        sampling_frac: float | None = None,
        seed: int | None = None,
        sampling_nquantiles: int = 100,
        xlims: tuple[float, float] | None = None,
        ylims: tuple[float, float] | None = None,
        marker_size: int = 60,
        marker_opacity: float = 0.1,
        interval_opacity: float = 0.5,
        interval_color: str = "grey",
        covered_color: str = "#009E73",
        uncovered_color: str = "#D55E00",
    ) -> alt.Chart:

        df = pd.DataFrame(
            {"y_pred": y_pred, "y_obs": y_obs, "y_lb": y_lb, "y_ub": y_ub}
        )

        if xlims is not None:
            df = df[(df["y_pred"] >= xlims[0]) & (df["y_pred"] <= xlims[1])]
            xscale = alt.Scale(domain=xlims)
        else:
            xscale = alt.Undefined

        if ylims is not None:
            df = df[(df["y_obs"] >= ylims[0]) & (df["y_obs"] <= ylims[1])]
            yscale = alt.Scale(domain=ylims)
        else:
            yscale = alt.Undefined

        if (sampling_frac is not None) and (seed is not None):
            df["quantile"] = qcut_with_tie_breaking(
                df["y_obs"], nbins=sampling_nquantiles, rng=np.random.default_rng(seed)
            )

            df_samples = df.groupby("quantile", observed=False).sample(
                frac=sampling_frac, replace=False, random_state=seed
            )
        elif (sampling_frac is not None) and (seed is None):
            raise ValueError("TODO")
        else:
            df_samples = df

        is_covered = (df_samples["y_lb"] <= df_samples["y_obs"]) & (
            df_samples["y_obs"] <= df_samples["y_ub"]
        )
        df_samples["is_covered"] = np.where(is_covered, "Covered", "Uncovered")

        if model_type == "freq":
            xtitle = "Predicted Number of Claims"
            ytitle = "Observed Number of Claims"
        elif model_type == "sev":
            xtitle = "Predicted Claim Amount"
            ytitle = "Observed Claim Amount"
        elif model_type == "loss":
            xtitle = "Predicted Loss on Policy"
            ytitle = "Observed Loss on Policy"
        else:
            raise ValueError("TODO")

        base = alt.Chart(df_samples).encode(
            x=alt.X("y_pred:Q", title="X"),
        )

        points = base.mark_point(size=marker_size, opacity=marker_opacity).encode(
            x=alt.X("y_pred:Q", title=xtitle, scale=xscale),
            y=alt.Y(
                "y_obs:Q",
                title=ytitle,
                scale=yscale,
            ),
            color=alt.Color(
                "is_covered:N",
                scale=alt.Scale(
                    # Labels of labeled/unlabeled points in legend:
                    domain=["Covered", "Uncovered"],
                    # Colors of labeled/unlabeled points:
                    range=[covered_color, uncovered_color],
                ),
                legend=alt.Legend(title="Validation Points", symbolOpacity=1),
            ),
            tooltip=[
                alt.Tooltip("y_pred:Q", title="X"),
                alt.Tooltip("y_obs:Q", title="y"),
            ],
        )

        interval = base.mark_area(opacity=interval_opacity).encode(
            x=alt.X("y_pred:Q", title=xtitle, scale=xscale),
            y=alt.Y("y_lb:Q", title=ytitle),
            y2="y_ub:Q",
            color=alt.value(interval_color),
        )

        # So points are drawn on-top of interval:
        plot = interval + points

        return plot

    return (plot_prediction_intervals_with_validation_points,)


@app.cell(hide_code=True)
def __(
    alpha,
    df_calib,
    df_valid,
    freq_valid_lb,
    freq_valid_pred,
    freq_valid_ub,
    plot_prediction_intervals_with_validation_points,
):
    plot_prediction_intervals_with_validation_points(
        y_pred=freq_valid_pred,
        y_obs=df_valid["ClaimNb"],
        y_lb=freq_valid_lb,
        y_ub=freq_valid_ub,
        model_type="freq",
        sampling_frac=0.05,
        seed=42,
        marker_opacity=0.1,
        xlims=(0, 0.7),
    ).properties(
        title=[
            "Conformally Calibrated Frequency Model",
            f"(Calibration Size = {len(df_calib):,}, Requested Coverage = {1-alpha:.1%})",
        ],
        height=300,
        width=400,
    )
    return


@app.cell
def __(compute_coverage, df_valid, freq_valid_lb, freq_valid_ub):
    freq_coverage_valid = compute_coverage(
        y_valid=df_valid["ClaimNb"], y_lb=freq_valid_lb, y_ub=freq_valid_ub
    )
    return (freq_coverage_valid,)


@app.cell(hide_code=True)
def __(df_calib, df_valid, freq_coverage_valid):
    print(
        f"{int(freq_coverage_valid*len(df_valid)):,} / "
        f"{len(df_valid):,} = {freq_coverage_valid:.2%} "
        "coverage on validation dataset "
        f"({len(df_calib):,} calibration samples)"
    )
    return


@app.cell
def __(np, qhat_sev, sev_valid_pred):
    sev_valid_lb = np.clip(sev_valid_pred - qhat_sev, a_min=0, a_max=np.inf)
    sev_valid_ub = sev_valid_pred + qhat_sev
    return sev_valid_lb, sev_valid_ub


@app.cell(hide_code=True)
def __(
    alpha,
    df_calib_claims,
    df_valid_claims,
    plot_prediction_intervals_with_validation_points,
    sev_valid_lb,
    sev_valid_pred,
    sev_valid_ub,
):
    plot_prediction_intervals_with_validation_points(
        y_pred=sev_valid_pred,
        y_obs=df_valid_claims["ClaimAmount"],
        y_lb=sev_valid_lb,
        y_ub=sev_valid_ub,
        model_type="sev",
        marker_opacity=0.2,
        xlims=(300, 8_000),
        ylims=(0, 100_000),
    ).properties(
        title=[
            "Conformally Calibrated Severity Model",
            f"(Calibration Size = {len(df_calib_claims):,}, Requested Coverage = {1-alpha:.1%})",
        ],
        height=300,
        width=400,
    )
    return


@app.cell
def __(compute_coverage, df_valid_claims, sev_valid_lb, sev_valid_ub):
    sev_coverage_valid = compute_coverage(
        y_valid=df_valid_claims["ClaimAmount"], y_lb=sev_valid_lb, y_ub=sev_valid_ub
    )
    return (sev_coverage_valid,)


@app.cell(hide_code=True)
def __(df_calib_claims, df_valid_claims, sev_coverage_valid):
    print(
        f"{int(sev_coverage_valid*len(df_valid_claims)):,} / "
        f"{len(df_valid_claims):,} = {sev_coverage_valid:.2%} "
        "coverage on validation dataset "
        f"({len(df_calib_claims):,} calibration samples)"
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Conformal Calibration of Individual Risk Loss Model""")
    return


@app.cell
def __(BaseEstimator, Pipeline, RegressorMixin, np, pd):
    class PolicyTotalLossModel(BaseEstimator, RegressorMixin):
        freq_model: Pipeline
        sev_model: Pipeline

        def __init__(self, freq_model: Pipeline, sev_model: Pipeline):
            self.freq_model = freq_model
            self.sev_model = sev_model

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            freq_pred = self.freq_model.predict(X)
            sev_pred = self.sev_model.predict(X)
            exposure = X["Exposure"].to_numpy()
            return exposure * freq_pred * sev_pred

    return (PolicyTotalLossModel,)


@app.cell
def __(PolicyTotalLossModel, model_freq, model_sev):
    model_loss = PolicyTotalLossModel(freq_model=model_freq, sev_model=model_sev)
    return (model_loss,)


@app.cell
def __(alpha, calibrate_regressor, df_calib, model_loss, np, seed_calib):
    loss_calib_pred = model_loss.predict(df_calib)
    qhat_loss = calibrate_regressor(
        y_pred=loss_calib_pred,
        y_true=df_calib["TotalLoss"],
        alpha=alpha,
        rng=np.random.default_rng(seed_calib),
    )
    return loss_calib_pred, qhat_loss


@app.cell
def __(df_valid, model_loss, np, qhat_loss):
    loss_valid_pred = model_loss.predict(df_valid)
    loss_valid_lb = np.clip(loss_valid_pred - qhat_loss, a_min=0, a_max=np.inf)
    loss_valid_ub = loss_valid_pred + qhat_loss
    return loss_valid_lb, loss_valid_pred, loss_valid_ub


@app.cell(hide_code=True)
def __(
    alpha,
    df_calib,
    df_valid,
    loss_valid_lb,
    loss_valid_pred,
    loss_valid_ub,
    plot_prediction_intervals_with_validation_points,
):
    plot_prediction_intervals_with_validation_points(
        y_pred=loss_valid_pred,
        y_obs=df_valid["TotalLoss"],
        y_lb=loss_valid_lb,
        y_ub=loss_valid_ub,
        model_type="loss",
        sampling_frac=0.05,
        seed=42,
        marker_opacity=0.2,
        xlims=(0, 1_000),
        ylims=(0, 20_000),
    ).properties(
        title=[
            "Conformally Calibrated Loss Model",
            f"(Calibration Size = {len(df_calib):,}, Requested Coverage = {1-alpha:.1%})",
        ],
        height=300,
        width=400,
    )
    return


@app.cell
def __(compute_coverage, df_valid, loss_valid_lb, loss_valid_ub):
    loss_coverage_valid = compute_coverage(
        y_valid=df_valid["TotalLoss"], y_lb=loss_valid_lb, y_ub=loss_valid_ub
    )
    return (loss_coverage_valid,)


@app.cell(hide_code=True)
def __(df_calib, df_valid, loss_coverage_valid):
    print(
        f"{int(loss_coverage_valid*len(df_valid)):,} / "
        f"{len(df_valid):,} = {loss_coverage_valid:.2%} "
        "coverage on validation dataset "
        f"({len(df_calib):,} calibration samples)"
    )
    return


if __name__ == "__main__":
    app.run()
