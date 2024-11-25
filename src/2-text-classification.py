import marimo

__generated_with = "0.9.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Applying Split Conformal Prediction to Text Classification""")
    return


@app.cell
def __():
    import typing
    import warnings
    from itertools import chain
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import optuna
    import pandas as pd
    import requests
    from scipy.stats import beta, betabinom
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import LabelEncoder

    return (
        LabelEncoder,
        Path,
        Pipeline,
        RandomForestClassifier,
        SelectKBest,
        TfidfVectorizer,
        accuracy_score,
        alt,
        beta,
        betabinom,
        chain,
        chi2,
        make_pipeline,
        mo,
        np,
        optuna,
        pd,
        requests,
        train_test_split,
        typing,
        warnings,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Loading and Inspecting the News Category Dataset""")
    return


@app.cell(hide_code=True)
def __(Path, mo, pd, requests):
    def load_news_classification_dataset(
        categories: list[str] | None = None,
        num_categories: int | None = None,
        save_dir: Path | str | None = None,
    ) -> pd.DataFrame:

        if (categories is not None) and (num_categories is not None):
            raise ValueError(
                "Both 'categories' and 'num_categories' were provided, "
                "but only one should be specified. Please provide either "
                "'categories' or 'num_categories', but not both."
            )

        json_path = download_news_classification_dataset(save_dir=save_dir)
        df = pd.read_json(json_path, lines=True)

        if num_categories is not None:
            category_counts: pd.Series = df.groupby("category")["category"].size()
            category_counts = category_counts.sort_values(ascending=False)
            categories = category_counts[:num_categories].index.to_list()

        if categories is not None:
            df = df[df["category"].isin(categories)]

        df["date"] = pd.to_datetime(df["date"])
        # Truncate nanoseconds to microseconds to prevent warnings
        # when displaying dataframe in marimo:
        df["date"] = df["date"].dt.floor("us")

        return df

    def download_news_classification_dataset(
        save_dir: Path | str | None = None, chunk_size: int = 8192
    ) -> Path:

        dataset_url = (
            "https://huggingface.co/datasets/heegyu/news-category-dataset/"
            "resolve/main/data.json?download=true"
        )
        dataset_filename = "news-category-dataset.json"

        if save_dir is None:
            save_dir = Path(".")
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        dataset_path = save_dir / dataset_filename

        if not dataset_path.exists():
            try:
                with requests.get(dataset_url, stream=True) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get("Content-length"))
                    with mo.status.progress_bar(
                        total=total_size,
                        title="Downloading `news-category-dataset.json`...",
                    ) as pbar:
                        with open(dataset_path, "wb") as file:
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                file.write(chunk)
                                pbar.update(len(chunk))
            except requests.RequestException as e:
                raise requests.RequestException(
                    "Failed to download `news-category-dataset` from `huggingface.co`. "
                    "If you're behind a corporate firewall, please manually download this "
                    f"dataset from {dataset_url} as save it as '{dataset_filename}' in the "
                    f"directory '{str(save_dir.absolute())}'."
                ) from e

        return dataset_path

    return (
        download_news_classification_dataset,
        load_news_classification_dataset,
    )


@app.cell
def __(LabelEncoder, load_news_classification_dataset):
    num_categories = 10
    data = load_news_classification_dataset(num_categories=num_categories)

    # Drop columns we won't analyse:
    data = data.drop(columns=["link", "authors"])

    # Combine `headline` and `short_description` into single text feature:
    data["X"] = data["headline"] + " " + data["short_description"]
    data = data.drop(columns=["headline", "short_description"])

    # Encode news categories as numerical values
    label_encoder = LabelEncoder().fit(y=data["category"])
    data["y"] = label_encoder.transform(data["category"])
    return data, label_encoder, num_categories


@app.cell(hide_code=True)
def __(data):
    data[["date", "category", "X", "y"]]
    return


@app.cell(hide_code=True)
def __(pd):
    def preview_news_dataset_label_distribution(
        data: pd.DataFrame, pretty_formatting: bool = True
    ) -> pd.DataFrame:
        df = data.groupby("category", observed=False).agg(
            count=("category", "size"),
        )
        df = df.sort_values("count", ascending=False)
        df["frac"] = df["count"] / df["count"].sum()
        df = df.reset_index(names="category")
        if pretty_formatting:
            df = df.transform(
                {
                    "category": lambda x: x,
                    # Display as comma-separated integers:
                    "count": lambda x: f"{x:,}",
                    # Display as percentages to 2 decimal points:
                    "frac": lambda x: f"{x:.2%}",
                }
            )
            df = df.rename(
                columns={
                    "category": "category",
                    "count": "Number of Samples",
                    "frac": "Percentage of Dataset",
                }
            )
        return df

    return (preview_news_dataset_label_distribution,)


@app.cell(hide_code=True)
def __(data, preview_news_dataset_label_distribution):
    preview_news_dataset_label_distribution(data)
    return


@app.cell
def __(np, pd, train_test_split):
    def train_test_calib_valid_rest_split(
        df: pd.DataFrame,
        train_size: int,
        test_size: int,
        calib_size: int,
        valid_size: int,
        rng: np.random.Generator,
        stratify: str,
    ) -> tuple[pd.DataFrame, ...]:

        try:
            assert train_size + test_size + calib_size + valid_size < len(df)
        except AssertionError:
            total = train_size + test_size + calib_size + valid_size
            raise ValueError(
                "Sum of `train_size`, `test_size`, `calib_size`, and `valid_size` "
                f"(= {total:,}) must be less than `len(df)` (= {len(df):,})."
            )

        df_train, df_not_train = train_test_split(
            df,
            train_size=train_size,
            stratify=df[stratify],
            random_state=sample_random_seed(rng),
        )
        df_test, df_not_test = train_test_split(
            df_not_train,
            train_size=test_size,
            stratify=df_not_train[stratify],
            random_state=sample_random_seed(rng),
        )
        df_calib, df_not_calib = train_test_split(
            df_not_test,
            train_size=calib_size,
            stratify=df_not_test[stratify],
            random_state=sample_random_seed(rng),
        )
        df_valid, df_rest = train_test_split(
            df_not_calib,
            train_size=valid_size,
            stratify=df_not_calib[stratify],
            random_state=sample_random_seed(rng),
        )
        return df_train, df_test, df_calib, df_valid, df_rest

    def sample_random_seed(rng: np.random.Generator) -> int:
        return rng.integers(low=0, high=np.iinfo(np.int32).max)

    return sample_random_seed, train_test_calib_valid_rest_split


@app.cell
def __():
    train_size = 8_000
    test_size = 2_000
    calib_size = 250
    valid_size = 50_000
    return calib_size, test_size, train_size, valid_size


@app.cell
def __(
    calib_size,
    data,
    np,
    test_size,
    train_size,
    train_test_calib_valid_rest_split,
    valid_size,
):
    rng_sample = np.random.default_rng(seed=42)
    df_train, df_test, df_calib, df_valid, df_rest = train_test_calib_valid_rest_split(
        data,
        train_size=train_size,
        test_size=test_size,
        calib_size=calib_size,
        valid_size=valid_size,
        rng=rng_sample,
        stratify="y",
    )
    return df_calib, df_rest, df_test, df_train, df_valid, rng_sample


@app.cell
def __(df_train, preview_news_dataset_label_distribution):
    preview_news_dataset_label_distribution(df_train)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Training a Simple Text Classification Model""")
    return


@app.cell
def __(
    Pipeline,
    RandomForestClassifier,
    SelectKBest,
    TfidfVectorizer,
    chi2,
    make_pipeline,
    np,
    sample_random_seed,
):
    def train_text_classification_model(
        X_train: np.ndarray, y_train: np.ndarray, rng: np.random.Generator, **params
    ) -> Pipeline:
        n_features = params.pop("n_features", 5000)
        model = make_pipeline(
            TfidfVectorizer(
                lowercase=True,
                strip_accents="ascii",
                stop_words="english",
            ),
            SelectKBest(score_func=chi2, k=n_features),
            RandomForestClassifier(random_state=sample_random_seed(rng), **params),
        )
        model.fit(X_train, y_train)
        return model

    return (train_text_classification_model,)


@app.cell
def __(
    accuracy_score,
    np,
    optuna,
    sample_random_seed,
    train_text_classification_model,
):
    def optimize_text_classification_model_hyperparameters(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        rng: np.random.Generator,
        ntrials: int = 10,
        verbose: bool = True,
    ) -> dict[str, int]:
        def objective_fn(trial: optuna.Trial) -> float:
            n_features = trial.suggest_int("n_features", 100, 10_000)
            randforest_params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
            model = train_text_classification_model(
                X_train=X_train,
                y_train=y_train,
                rng=rng,
                n_features=n_features,
                **randforest_params,
            )
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred)

        current_verbosity = optuna.logging.get_verbosity()
        sampler = optuna.samplers.TPESampler(seed=sample_random_seed(rng))
        try:
            optuna.logging.set_verbosity(
                optuna.logging.WARNING if not verbose else current_verbosity
            )
            study = optuna.create_study(sampler=sampler, direction="maximize")
            study.optimize(objective_fn, n_trials=ntrials, show_progress_bar=verbose)
        finally:
            optuna.logging.set_verbosity(current_verbosity)

        return study.best_params

    return (optimize_text_classification_model_hyperparameters,)


@app.cell
def __(Pipeline, np):
    def calibrate_model(
        model: Pipeline, X_calib: np.ndarray, y_calib: np.ndarray, alpha: float
    ) -> np.ndarray:

        # Compute non-conformity scores on calibration dataset:
        logits = model.predict_proba(X_calib)
        scores = 1 - np.take_along_axis(
            arr=logits, indices=y_calib.reshape(-1, 1), axis=1
        ).reshape(-1)

        # Compute (1-alpha) quantile of scores:
        scores = np.sort(scores, axis=-1)
        calib_size = y_calib.size
        qhat_idx = np.ceil((calib_size + 1) * (1 - alpha)).astype(int) - 1
        qhat = np.append(scores, [1])[qhat_idx]

        return qhat

    return (calibrate_model,)


@app.cell
def __(np):
    def compute_prediction_mask(
        logits: np.ndarray,
        qhat: np.ndarray,
    ) -> np.ndarray:
        return 1 - logits <= qhat

    return (compute_prediction_mask,)


@app.cell
def __(LabelEncoder, np):
    def compute_prediction_set(
        prediction_mask: np.ndarray,
        label_encoder: LabelEncoder | None = None,
    ) -> np.ndarray:
        prediction_sets: list[list[str]] = []
        for mask in prediction_mask:
            idx = np.argwhere(mask).reshape(-1)
            prediction_sets.append(label_encoder.inverse_transform(idx).tolist())
        return prediction_sets

    return (compute_prediction_set,)


@app.cell
def __(
    df_test,
    df_train,
    np,
    optimize_text_classification_model_hyperparameters,
):
    rng_optimize = np.random.default_rng(seed=123)
    best_params = optimize_text_classification_model_hyperparameters(
        X_train=df_train["X"],
        y_train=df_train["y"],
        X_test=df_test["X"],
        y_test=df_test["y"],
        rng=rng_optimize,
    )
    return best_params, rng_optimize


@app.cell
def __(best_params, df_train, np, train_text_classification_model):
    rng_train = np.random.default_rng(seed=456)
    model = train_text_classification_model(
        X_train=df_train["X"], y_train=df_train["y"], rng=rng_train, **best_params
    )
    return model, rng_train


@app.cell
def __(accuracy_score, df_test, df_train, model):
    accuracy_train = accuracy_score(
        y_pred=model.predict(df_train["X"]), y_true=df_train["y"]
    )
    accuracy_test = accuracy_score(
        y_pred=model.predict(df_test["X"]), y_true=df_test["y"]
    )
    return accuracy_test, accuracy_train


@app.cell(hide_code=True)
def __(accuracy_test, accuracy_train):
    print(f"Point Prediction Accuracy on Training Set: {accuracy_train:.2%}")
    print(f"Point Prediction Accuracy on Test Set: {accuracy_test:.2%}")
    return


@app.cell
def __(calibrate_model, df_calib, model):
    alpha = 0.05
    qhat = calibrate_model(
        model=model,
        X_calib=df_calib["X"],
        y_calib=df_calib["y"].to_numpy(),
        alpha=alpha,
    )
    return alpha, qhat


@app.cell
def __(df_valid, model, np):
    logits_valid = model.predict_proba(df_valid["X"])
    y_pred_valid = np.argmax(logits_valid, axis=1)
    return logits_valid, y_pred_valid


@app.cell
def __(compute_prediction_mask, logits_valid, qhat):
    pred_masks_valid = compute_prediction_mask(logits=logits_valid, qhat=qhat)
    return (pred_masks_valid,)


@app.cell
def __(compute_prediction_set, label_encoder, pred_masks_valid):
    pred_sets_valid = compute_prediction_set(
        prediction_mask=pred_masks_valid, label_encoder=label_encoder
    )
    return (pred_sets_valid,)


@app.cell(hide_code=True)
def __(LabelEncoder, compute_prediction_set, np, pd):
    def summarize_prediction_sets(
        X: np.ndarray,
        logits: np.ndarray,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        prediction_mask: np.ndarray,
        label_encoder: LabelEncoder,
        logits_dp: int = 3,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "X": X,
                "logits": np.round(logits, decimals=logits_dp).tolist(),
                "y_pred": label_encoder.inverse_transform(y_pred),
                "y_true": label_encoder.inverse_transform(y_true),
            }
        )
        df["pred_set"] = compute_prediction_set(
            prediction_mask=prediction_mask, label_encoder=label_encoder
        )
        df["pred_set_size"] = np.sum(prediction_mask, axis=1)
        df["y_true_in_set"] = df.apply(
            lambda row: row["y_true"] in row["pred_set"], axis=1
        )
        df = df[
            [
                "X",
                "logits",
                "y_pred",
                "pred_set",
                "y_true",
                "pred_set_size",
                "y_true_in_set",
            ]
        ]
        return df

    return (summarize_prediction_sets,)


@app.cell(hide_code=True)
def __(
    df_valid,
    label_encoder,
    logits_valid,
    pred_masks_valid,
    summarize_prediction_sets,
    y_pred_valid,
):
    df_summary = summarize_prediction_sets(
        X=df_valid["X"],
        logits=logits_valid,
        y_pred=y_pred_valid,
        y_true=df_valid["y"],
        prediction_mask=pred_masks_valid,
        label_encoder=label_encoder,
    )
    return (df_summary,)


@app.cell(hide_code=True)
def __(df_summary):
    df_summary.rename(
        columns={
            "X": "Text",
            "logits": "Logits",
            "y_pred": "Point Prediction",
            "pred_set": "Prediction Set",
            "y_true": "True Label",
            "pred_set_size": "Prediction Set Size",
            "y_true_in_set": "True Label in Set?",
        },
        inplace=False,
    )
    return


@app.cell
def __(accuracy_score, df_summary, df_valid, y_pred_valid):
    accuracy_valid = accuracy_score(y_pred=y_pred_valid, y_true=df_valid["y"])
    total_covered_valid = df_summary["y_true_in_set"].sum()
    coverage_valid = total_covered_valid / len(df_summary)
    return accuracy_valid, coverage_valid, total_covered_valid


@app.cell(hide_code=True)
def __(
    accuracy_valid,
    alpha,
    coverage_valid,
    df_summary,
    total_covered_valid,
):
    print(f"Point Prediction Accuracy on Validation Set: {accuracy_valid:.2%}")
    print(
        f"Prediction Set Coverage (𝛼 = {alpha}) "
        f"on Validation Set ({total_covered_valid:,} / {len(df_summary):,}): "
        f" {coverage_valid:.2%}"
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Sampling from the Empirical Coverage Distribution""")
    return


@app.cell(hide_code=True)
def __(alt, np, pd):
    def plot_article_date_distribution(dates: pd.Series, categories: pd.Series):
        df = pd.DataFrame({"date": dates, "category": categories.str.title()})
        df["month"] = df["date"].dt.to_period("M")
        df_monthly = df.groupby(["month", "category"]).size().reset_index(name="count")
        df_monthly_totals = (
            df_monthly.groupby("month")["count"].sum().reset_index(name="total")
        )
        df_monthly_totals["cumsum"] = np.cumsum(df_monthly_totals["total"])

        df_monthly = df_monthly.merge(df_monthly_totals, on="month")
        df_monthly["frac"] = df_monthly["count"] / df_monthly["total"]

        # Altair can't plot `pd.Period` objects; convert back to timestamp:
        df_monthly_totals["month"] = df_monthly_totals["month"].dt.to_timestamp()
        df_monthly["month"] = df_monthly["month"].dt.to_timestamp()

        num_articles_plot = (
            alt.Chart(df_monthly_totals)
            .mark_area()
            .encode(
                x=alt.X("month:T", title="Date"),  # Adjust maxbins for bin granularity
                y=alt.Y("cumsum:Q", title="Cumulative Number of Articles"),
            )
            .properties(title="Cumulative Number of Articles Published over Time")
        )

        category_area_plot = (
            alt.Chart(df_monthly)
            .mark_area()
            .encode(
                x=alt.X("month:T", title="Date"),
                y=alt.Y("frac:Q", stack="normalize", title="Fraction of Categories"),
                color=alt.Color("category:N", title="Category"),
            )
            .properties(title="Fraction of News Categories over Time")
        )

        return num_articles_plot | category_area_plot

    return (plot_article_date_distribution,)


@app.cell
def __(data, plot_article_date_distribution):
    plot_article_date_distribution(dates=data["date"], categories=data["category"])
    return


@app.cell
def __(np):
    def compute_coverage(prediction_mask: np.ndarray, y_true: np.ndarray) -> float:
        is_covered = np.take_along_axis(
            arr=prediction_mask, indices=y_true.reshape(-1, 1), axis=1
        ).reshape(-1)
        return np.mean(is_covered).item()

    return (compute_coverage,)


@app.cell
def __(
    Pipeline,
    calibrate_model,
    compute_coverage,
    compute_prediction_mask,
    mo,
    np,
    pd,
    typing,
):
    def sample_empirical_coverage_distribution(
        num_samples: int,
        calib_size: int,
        model: Pipeline,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        alpha: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:

        logits_valid = model.predict_proba(X_valid)

        results: list[dict[str, typing.Any]] = []
        for batch_idx in mo.status.progress_bar(
            range(num_samples),
            total=num_samples,
            title="Sampling Empirical Coverages...",
        ):

            batch_slice = batch_idx * calib_size + np.arange(calib_size)

            qhat = calibrate_model(
                model=model,
                X_calib=X_calib[batch_slice],
                y_calib=y_calib[batch_slice],
                alpha=alpha,
            )

            pred_mask_valid = compute_prediction_mask(logits=logits_valid, qhat=qhat)

            result_i = {
                "coverage": compute_coverage(
                    prediction_mask=pred_mask_valid, y_true=y_valid
                ),
                "avg_set_size": np.mean(np.sum(pred_mask_valid, axis=1)),
            }

            results.append(result_i)

        return pd.DataFrame.from_records(results)

    return (sample_empirical_coverage_distribution,)


@app.cell
def __():
    params_covdist = {"num_samples": 1_000, "calib_size": 100, "alpha": 0.1}
    return (params_covdist,)


@app.cell
def __(
    df_calib,
    df_rest,
    df_valid,
    np,
    params_covdist,
    pd,
    sample_random_seed,
    train_test_split,
):
    rng_covdist = np.random.default_rng(1024)
    df_pool_covdist = pd.concat([df_calib, df_valid, df_rest], axis=0)
    df_calib_covdist, df_valid_covdist = train_test_split(
        df_pool_covdist,
        train_size=params_covdist["num_samples"] * params_covdist["calib_size"],
        random_state=sample_random_seed(rng_covdist),
        stratify=df_pool_covdist["y"],
    )
    return df_calib_covdist, df_pool_covdist, df_valid_covdist, rng_covdist


@app.cell
def __(
    df_calib_covdist,
    df_valid_covdist,
    model,
    params_covdist,
    rng_covdist,
    sample_empirical_coverage_distribution,
):
    df_covdist = sample_empirical_coverage_distribution(
        num_samples=params_covdist["num_samples"],
        calib_size=params_covdist["calib_size"],
        model=model,
        X_calib=df_calib_covdist["X"].to_numpy(),
        y_calib=df_calib_covdist["y"].to_numpy(),
        X_valid=df_valid_covdist["X"].to_numpy(),
        y_valid=df_valid_covdist["y"].to_numpy(),
        alpha=params_covdist["alpha"],
        rng=rng_covdist,
    )
    return (df_covdist,)


@app.cell(hide_code=True)
def __(betabinom, np, warnings):
    def theoretical_finite_sample_coverage_pmf(
        covered_size: np.ndarray,
        alpha: np.ndarray,
        calib_size: np.ndarray,
        valid_size: np.ndarray,
    ) -> np.ndarray:
        is_alpha_too_small = alpha < 1 / (calib_size + 1)
        if np.any(is_alpha_too_small):
            warnings.warn(
                "One or more `alpha` is smaller than `1/(calib_size+1)`. "
                "Prediction intervals for these `alpha` are infinitely larg, "
                "so the theoretical coverage distribution no longer applies."
            )
        a = np.ceil((1 - alpha) * (calib_size + 1))
        b = np.floor(alpha * (calib_size + 1))
        return betabinom.pmf(covered_size, n=valid_size, a=a, b=b)

    return (theoretical_finite_sample_coverage_pmf,)


@app.cell(hide_code=True)
def __(alt, np, pd, theoretical_finite_sample_coverage_pmf):
    def plot_observed_coverage_vs_theory(
        observed_coverage: np.ndarray,
        calib_size: int,
        valid_size: int,
        alpha: float,
        hist_nbins: int = 20,
        hist_strokewidth: int = 1,
        hist_strokecolor: str = "grey",
        line_npoints: int = 500,
    ) -> alt.Chart:

        hist_values, bin_edges = np.histogram(
            observed_coverage, bins=hist_nbins, density=True
        )
        df_hist = pd.DataFrame(
            {
                "bin_start": bin_edges[1:],
                "bin_end": bin_edges[:-1],
                "density": hist_values,
                "key": "Observed",
            }
        )
        hist = (
            alt.Chart(df_hist)
            .mark_bar(
                stroke=hist_strokecolor,
                strokeWidth=hist_strokewidth,
            )
            .encode(
                alt.X("bin_start:Q", title="Coverage Fraction", bin="binned"),
                alt.X2("bin_end:Q"),
                alt.Y("sum(density):Q", title="Probability Density"),
                color=alt.Color("key:N", title="Legend"),
                tooltip=[
                    alt.Tooltip("bin_start:Q", title="Bin Start", format=".3f"),
                    alt.Tooltip("bin_end:Q", title="Bin End", format=".3f"),
                    alt.Tooltip("density:Q", title="Probability Density", format=".3f"),
                ],
            )
        )

        x_frac = np.linspace(start=0, stop=1, num=line_npoints)
        x_int = np.unique(np.round(valid_size * x_frac, decimals=0).astype(int))
        pmf_int = theoretical_finite_sample_coverage_pmf(
            covered_size=x_int,
            alpha=alpha,
            calib_size=calib_size,
            valid_size=valid_size,
        )
        pdf_frac = pmf_int * valid_size
        df_pdf = pd.DataFrame({"x": x_frac, "pdf": pdf_frac, "key": "Theory"})
        line = (
            alt.Chart(df_pdf)
            .mark_line()
            .encode(
                x=alt.X("x:Q", title="Coverage Fraction"),
                y=alt.Y("pdf:Q", title="Probability Density"),
                color=alt.Color("key:N", title="Legend"),
            )
        )

        graph = hist + line
        graph = graph.properties(
            title=[
                f"Histogram of Observed Coverage ({len(observed_coverage)} Samples) vs Theory",
                f"(𝛼 = {alpha:.1%}, Calibration Size = {calib_size:,}, Validation Size = {valid_size:,})",
            ]
        )

        return graph

    return (plot_observed_coverage_vs_theory,)


@app.cell(hide_code=True)
def __(
    df_covdist,
    df_valid_covdist,
    params_covdist,
    plot_observed_coverage_vs_theory,
):
    plot_observed_coverage_vs_theory(
        observed_coverage=df_covdist["coverage"],
        calib_size=params_covdist["calib_size"],
        valid_size=len(df_valid_covdist),
        alpha=params_covdist["alpha"],
    ).properties(width=500, height=300)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Class-Conditional Coverage""")
    return


@app.cell
def __(Any, compute_coverage, np):
    def compute_class_conditional_coverage(
        prediction_mask: np.ndarray, y_true: np.ndarray, labels: list[Any]
    ) -> np.ndarray:
        coverages: list[float] = []
        for label in labels:
            label_mask = y_true == label
            coverages.append(
                compute_coverage(
                    prediction_mask=prediction_mask[label_mask, :],
                    y_true=y_true[label_mask],
                )
            )
        return np.array(coverages)

    return (compute_class_conditional_coverage,)


@app.cell
def __():
    params_labelcond = {"alpha": 0.1, "calib_size": 20_000}
    return (params_labelcond,)


@app.cell
def __(
    df_calib,
    df_rest,
    df_valid,
    np,
    params_labelcond,
    pd,
    sample_random_seed,
    train_test_split,
):
    rng_labelcond = np.random.default_rng(1024)
    df_pool_labelcond = pd.concat([df_calib, df_valid, df_rest], axis=0)
    df_calib_labelcond, df_valid_labelcond = train_test_split(
        df_pool_labelcond,
        train_size=params_labelcond["calib_size"],
        random_state=sample_random_seed(rng_labelcond),
        stratify=df_pool_labelcond["y"],
    )
    return (
        df_calib_labelcond,
        df_pool_labelcond,
        df_valid_labelcond,
        rng_labelcond,
    )


@app.cell
def __(calibrate_model, df_calib_labelcond, model, params_labelcond):
    qhat_uncond = calibrate_model(
        model=model,
        X_calib=df_calib_labelcond["X"].to_numpy(),
        y_calib=df_calib_labelcond["y"].to_numpy(),
        alpha=params_labelcond["alpha"],
    )
    return (qhat_uncond,)


@app.cell
def __(df_valid_labelcond, model):
    logits_valid_labelcond = model.predict_proba(df_valid_labelcond["X"])
    return (logits_valid_labelcond,)


@app.cell
def __(
    compute_class_conditional_coverage,
    compute_prediction_mask,
    df_valid_labelcond,
    logits_valid_labelcond,
    np,
    num_categories,
    qhat_uncond,
):
    pred_mask_uncond = compute_prediction_mask(
        logits=logits_valid_labelcond, qhat=qhat_uncond
    )
    coverage_qhat_uncond = compute_class_conditional_coverage(
        prediction_mask=pred_mask_uncond,
        y_true=df_valid_labelcond["y"].to_numpy(),
        labels=np.arange(num_categories),
    )
    return coverage_qhat_uncond, pred_mask_uncond


@app.cell
def __(df_calib_labelcond, np):
    label_sizes_calib: np.ndarray = df_calib_labelcond.groupby("y").size().to_numpy()
    label_sizes_calib
    return (label_sizes_calib,)


@app.cell
def __(df_valid_labelcond, np):
    label_sizes_valid: np.ndarray = df_valid_labelcond.groupby("y").size().to_numpy()
    label_sizes_valid
    return (label_sizes_valid,)


@app.cell(hide_code=True)
def __(Any, alt, chain, np, pd):
    def plot_label_conditional_quantity(
        labels: list[Any],
        quantity_name: str,
        uncond_quantity: np.ndarray | None = None,
        cond_quantity: np.ndarray | None = None,
        calib_sizes: list[int] | None = None,
        valid_sizes: list[int] | None = None,
        hline_yposition: float | None = None,
        hline_color: str = "limegreen",
        hline_strokewidth: float = 2.0,
        hline_strokedash: tuple[int, int] = (10, 2),
        bar_text_color: str = "grey",
        bar_text_size: int = 12,
        bar_text_ypadding: float = 0.05,
        bar_text_first_dy: float = -15.0,
        bar_text_second_dy: float = 0.0,
        uncond_quantity_label="Label Unconditional Calibration",
        cond_quantity_label="Label Conditional Calibration",
        quantity_format_str: str | alt.Undefined = alt.Undefined,
    ) -> alt.Chart:

        if (uncond_quantity is None) and (cond_quantity is None):
            raise ValueError(
                "Both 'uncond_quantity' and 'cond_quantity' are missing. "
                "Please specify at least one of these quantities to proceed."
            )
        two_bar_sets = (uncond_quantity is not None) and (cond_quantity is not None)

        if two_bar_sets:
            labels_both = chain.from_iterable(2 * [labels])
            values_both = uncond_quantity.tolist() + cond_quantity.tolist()
            groups_both = (len(labels) * [uncond_quantity_label]) + (
                len(labels) * [cond_quantity_label]
            )
            df = pd.DataFrame(
                {"label": labels_both, "values": values_both, "group": groups_both}
            )
        else:
            df = pd.DataFrame(
                {
                    "label": labels,
                    "values": uncond_quantity
                    if uncond_quantity is not None
                    else cond_quantity,
                }
            )

        if two_bar_sets:
            bar_color = alt.Color(
                "group:N",
                title="Legend",
                # Specify order of bars:
                scale=alt.Scale(domain=[uncond_quantity_label, cond_quantity_label]),
            )
            x_offset = alt.XOffset(
                "group:N",
                # Specify order of bars:
                scale=alt.Scale(domain=[uncond_quantity_label, cond_quantity_label]),
            )
        else:
            bar_color = alt.Undefined
            x_offset = alt.Undefined

        tooltips = [
            alt.Tooltip("label:N", title="Label"),
            alt.Tooltip(
                "values:Q", title=quantity_name.title(), format=quantity_format_str
            ),
        ]
        if two_bar_sets:
            tooltips.append(alt.Tooltip("group:N", title="Legend"))

        graph = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("label:N", title="Label", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("values:Q", title=quantity_name.title()),
                color=bar_color,
                xOffset=x_offset,
                tooltip=tooltips,
            )
        )

        cond_max = np.max(cond_quantity) if cond_quantity is not None else np.nan
        uncond_max = np.max(uncond_quantity) if uncond_quantity is not None else np.nan
        y_max = np.nanmax(np.append([cond_max], [uncond_max]))
        sizes_dict = {"calib": calib_sizes, "valid": valid_sizes}
        is_first_text_row = True
        for sizes_name, sizes in sizes_dict.items():
            if sizes is not None:
                df_sizes = pd.DataFrame(
                    {
                        "label": labels,
                        "values": y_max + bar_text_ypadding,
                        "size": [f"{sizes_name.title()}: {s:,}" for s in sizes],
                    }
                )
                if is_first_text_row:
                    dy = bar_text_first_dy
                else:
                    dy = bar_text_second_dy
                bar_labels = (
                    alt.Chart(df_sizes)
                    .mark_text(
                        align="center",  # Center text horizontally
                        baseline="top",  # Position text above each bar
                        dy=dy,  # Offset text to be slightly above each bar
                        size=bar_text_size,
                        color=bar_text_color,
                    )
                    .encode(
                        x=alt.X("label:N"),  # Same x encoding as bars
                        y=alt.Y("values:Q"),  # Same y encoding as bars
                        text=alt.Text("size:N"),
                    )
                )
                graph = graph + bar_labels
                is_first_text_row = False

        if hline_yposition is not None:
            df_hline = pd.DataFrame(
                {
                    "y": [hline_yposition],
                }
            )
            hline = (
                alt.Chart(df_hline)
                .mark_rule(
                    strokeWidth=hline_strokewidth,
                    strokeDash=hline_strokedash,
                )
                .encode(y=alt.Y("y:Q"), color=alt.value(hline_color))
            )
            graph = graph + hline

        return graph

    return (plot_label_conditional_quantity,)


@app.cell(hide_code=True)
def __(
    coverage_qhat_uncond,
    label_encoder,
    label_sizes_calib,
    label_sizes_valid,
    params_labelcond,
    plot_label_conditional_quantity,
):
    plot_label_conditional_quantity(
        uncond_quantity=coverage_qhat_uncond,
        labels=[c.title() for c in label_encoder.classes_],
        calib_sizes=label_sizes_calib,
        valid_sizes=label_sizes_valid,
        hline_yposition=1 - params_labelcond["alpha"],
        quantity_name="Coverage",
        quantity_format_str=".2%",
    ).properties(title="Label Conditional Coverage", height=400, width=800)
    return


@app.cell
def __(Pipeline, calibrate_model, np):
    def calibrate_model_labelcond(
        model: Pipeline,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        alpha: float,
        labels: np.ndarray,
    ) -> np.ndarray:
        qhats: list[float] = []
        for label in labels:
            label_mask = y_calib == label
            qhats.append(
                calibrate_model(
                    model=model,
                    X_calib=X_calib[label_mask],
                    y_calib=y_calib[label_mask],
                    alpha=alpha,
                )
            )
        return np.array(qhats)

    return (calibrate_model_labelcond,)


@app.cell
def __(
    calibrate_model_labelcond,
    df_calib_labelcond,
    model,
    np,
    num_categories,
    params_labelcond,
):
    qhat_cond = calibrate_model_labelcond(
        model=model,
        X_calib=df_calib_labelcond["X"].to_numpy(),
        y_calib=df_calib_labelcond["y"].to_numpy(),
        alpha=params_labelcond["alpha"],
        labels=np.arange(num_categories),
    )
    return (qhat_cond,)


@app.cell
def __(
    compute_class_conditional_coverage,
    compute_prediction_mask,
    df_valid_labelcond,
    logits_valid_labelcond,
    np,
    num_categories,
    qhat_cond,
):
    pred_mask_cond = compute_prediction_mask(
        logits=logits_valid_labelcond, qhat=qhat_cond
    )
    coverage_qhat_cond = compute_class_conditional_coverage(
        prediction_mask=pred_mask_cond,
        y_true=df_valid_labelcond["y"].to_numpy(),
        labels=np.arange(num_categories),
    )
    return coverage_qhat_cond, pred_mask_cond


@app.cell(hide_code=True)
def __(
    coverage_qhat_cond,
    coverage_qhat_uncond,
    label_encoder,
    label_sizes_calib,
    label_sizes_valid,
    params_labelcond,
    plot_label_conditional_quantity,
):
    plot_label_conditional_quantity(
        uncond_quantity=coverage_qhat_uncond,
        cond_quantity=coverage_qhat_cond,
        labels=[c.title() for c in label_encoder.classes_],
        calib_sizes=label_sizes_calib,
        valid_sizes=label_sizes_valid,
        hline_yposition=1 - params_labelcond["alpha"],
        quantity_name="Coverage",
        quantity_format_str=".2%",
    ).properties(
        title=(
            "Label Conditional Coverage for Label Unconditional "
            "Calibration vs Label Conditional Calibration"
        ),
        height=400,
        width=800,
    )
    return


@app.cell(hide_code=True)
def __(
    label_encoder,
    label_sizes_calib,
    np,
    num_categories,
    plot_label_conditional_quantity,
    qhat_cond,
    qhat_uncond,
):
    plot_label_conditional_quantity(
        uncond_quantity=np.repeat(1 - qhat_uncond, repeats=num_categories),
        cond_quantity=1 - qhat_cond,
        labels=[c.title() for c in label_encoder.classes_],
        calib_sizes=label_sizes_calib,
        hline_yposition=1 - qhat_uncond,
        quantity_name="Logits Conformal Threshold",
        quantity_format_str=".2%",
        bar_text_ypadding=0.005,
    ).properties(
        title=[
            "Logit Thresholds for Class Unconditional",
            "vs Class Conditional Split Conformal Prediction",
        ],
        height=400,
        width=800,
    )
    return


@app.cell
def __(Any, np):
    def compute_avg_set_size(prediction_mask: np.ndarray) -> float:
        return np.mean(np.sum(prediction_mask, axis=1)).item()

    def compute_label_conditional_avg_set_size(
        prediction_mask: np.ndarray, y_true: np.ndarray, labels: list[Any]
    ) -> np.ndarray:
        avg_set_sizes: list[float] = []
        for label in labels:
            label_mask = y_true == label
            avg_set_sizes.append(compute_avg_set_size(prediction_mask[label_mask, :]))
        return np.array(avg_set_sizes)

    return compute_avg_set_size, compute_label_conditional_avg_set_size


@app.cell
def __(
    compute_label_conditional_avg_set_size,
    df_valid_labelcond,
    np,
    num_categories,
    pred_mask_cond,
    pred_mask_uncond,
):
    avgsetsize_quncond = compute_label_conditional_avg_set_size(
        prediction_mask=pred_mask_uncond,
        y_true=df_valid_labelcond["y"].to_numpy(),
        labels=np.arange(num_categories),
    )
    avgsetsize_qcond = compute_label_conditional_avg_set_size(
        prediction_mask=pred_mask_cond,
        y_true=df_valid_labelcond["y"].to_numpy(),
        labels=np.arange(num_categories),
    )
    return avgsetsize_qcond, avgsetsize_quncond


@app.cell(hide_code=True)
def __(
    avgsetsize_qcond,
    avgsetsize_quncond,
    label_encoder,
    label_sizes_calib,
    label_sizes_valid,
    plot_label_conditional_quantity,
):
    plot_label_conditional_quantity(
        uncond_quantity=avgsetsize_quncond,
        cond_quantity=avgsetsize_qcond,
        labels=[c.title() for c in label_encoder.classes_],
        calib_sizes=label_sizes_calib,
        valid_sizes=label_sizes_valid,
        quantity_name="Average Set Size over Validation Dataset",
        bar_text_ypadding=0.3,
    ).properties(
        title=[
            "Average Prediction Set Sizes for Class Unconditional",
            "vs Class Conditional Split Conformal Prediction",
        ],
        height=400,
        width=800,
    )
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
