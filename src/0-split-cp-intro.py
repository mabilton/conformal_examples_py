import marimo

__generated_with = "0.9.16"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        r"""
        # Introduction to Split Conformal Prediction

        This notebook.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## What is Split Conformal Prediction?

        *Conformal Prediction* is a
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## What is Meant by 'Guaranteed Coverage'?

        Conformal Prediction promises '' ; but what does this promise actually mean?



        A naive interpretation of this claim is something along the lines of "Conformal Prediction will". Unfortunately, this interpretation is *completely wrong*.

        There are two caveats:

        1. TODO.
        2. The guaranteed covarge is over *all possible*, meaning that the coverage
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Calibration Set Variability

        It's entirely possible that purely by chance that we sample a calibration set that our model does particularly well at predicting relative to the population distribution, meaning we will 'underestimate' the true distribution of the non-conformity scores.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Marginal vs Conditional Coverage


        $$
        P(y_{n+1} \in C_\alpha) = \mathbb{E}_{(x, y) \sim p(x,y)} \big[ \mathbb{1}(y) \big]
        $$

        $$
        1-\alpha \leq P(y_{n+1} \in C) \leq 1-\alpha+\frac{1}{n_{\text{calib}}+1}
        $$
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline, Pipeline
    from scipy.stats import beta, norm, betabinom
    from scipy.special import factorial
    import pandas as pd
    import altair as alt
    import warnings
    return (
        LinearRegression,
        Pipeline,
        PolynomialFeatures,
        alt,
        beta,
        betabinom,
        factorial,
        make_pipeline,
        mo,
        norm,
        np,
        pd,
        warnings,
    )


@app.cell
def __(beta, np, warnings):
    def theoretical_infinite_sample_coverage_pdf(covered_frac: np.ndarray, alpha: np.ndarray, calib_size: np.ndarray) -> np.ndarray:
        is_alpha_too_small = alpha < 1 / (calib_size+1)
        if np.any(is_alpha_too_small):
            warnings.warn(
                'One or more `alpha` is smaller than `1/(calib_size+1)`. '
                'Prediction intervals for these `alpha` are infinitely large, '
                'so the theoretical coverage distribution no longer applies.'
            )
        a = np.ceil((1 - alpha) * (calib_size + 1))
        b = np.floor(alpha * (calib_size + 1))
        return beta.pdf(covered_frac, a=a, b=b)
    return (theoretical_infinite_sample_coverage_pdf,)


@app.cell
def __(betabinom, np, warnings):
    def theoretical_finite_sample_coverage_pdf(covered_size: np.ndarray, alpha: np.ndarray, calib_size: np.ndarray, valid_size: np.ndarray) -> np.ndarray:
        is_alpha_too_small = alpha < 1 / (calib_size+1)
        if np.any(is_alpha_too_small):
            warnings.warn(
                'One or more `alpha` is smaller than `1/(calib_size+1)`. '
                'Prediction intervals for these `alpha` are infinitely larg, '
                'so the theoretical coverage distribution no longer applies.'
            )
        a = np.ceil((1 - alpha) * (calib_size + 1))
        b = np.floor(alpha * (calib_size + 1))
        return betabinom.pmf(covered_size, n=valid_size, a=a, b=b)
    return (theoretical_finite_sample_coverage_pdf,)


@app.cell(hide_code=True)
def __(alt, np, pd, theoretical_finite_sample_coverage_pdf):
    def plot_theoretical_finite_sample_coverage_pdf(alpha: float, calib_size: int, valid_size: int) -> alt.Chart:
        k_values = np.arange(0, valid_size + 1)
        probabilities = [theoretical_finite_sample_coverage_pdf(k, alpha=alpha, valid_size=valid_size,  calib_size=calib_size) for k in k_values]    
        df = pd.DataFrame({
            'k': k_values,
            'Probability': probabilities
        })
        kmax = df['k'].max()
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(
                'k:Q', 
                title='Number of Validation Samples Contained in Prediction Set',
                scale=alt.Scale(domain=[0, kmax]),
            ),
            y=alt.Y('Probability:Q', title='Probability Mass'),
            tooltip=[alt.Tooltip('k:Q', title='Number of Samples'), alt.Tooltip('Probability:Q', title='Probability Mass')]
        ).properties(
            title=[
                f'Theoretical Coverage Distribution for α = {alpha:.1%}, ',
                f'{calib_size:,} Calibration Points, and '
                f'{valid_size:,} Validation Points'
            ],
        )
        return chart
    return (plot_theoretical_finite_sample_coverage_pdf,)


@app.cell(hide_code=True)
def __(alt, np, pd, theoretical_infinite_sample_coverage_pdf):
    def plot_theoretical_infinite_sample_coverage_pdf(alpha: float, calib_size: int) -> alt.Chart:
        covered_frac = np.linspace(0, 1, 500)
        pdf_values = theoretical_infinite_sample_coverage_pdf(covered_frac, alpha=alpha, calib_size=calib_size)    
        df = pd.DataFrame({
            'covered_frac': covered_frac,
            'pdf': pdf_values
        }) 
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X('covered_frac:Q', title='Coverage', axis=alt.Axis(format=".0%")),
            y=alt.Y('pdf:Q', title='Probability Density'),
        ).properties(
            title=[
                f'Theoretical Coverage Distribution for α = {alpha:.1%}, ',
                f'{calib_size:,} Calibration Points, and '
                'Infinitely Many Validation Points'
            ]
        )
        return chart
    return (plot_theoretical_infinite_sample_coverage_pdf,)


@app.cell
def __():
    plot_params_1 = {
        'alpha': 0.1,
        'calib_size': 50,
        'valid_size': 50
    }
    return (plot_params_1,)


@app.cell(hide_code=True)
def __(
    plot_params_1,
    plot_theoretical_finite_sample_coverage_pdf,
    plot_theoretical_infinite_sample_coverage_pdf,
):
    plot_theoretical_finite_sample_coverage_pdf(
        alpha=plot_params_1['alpha'], 
        calib_size=plot_params_1['calib_size'], 
        valid_size=plot_params_1['valid_size']
    ) | \
    plot_theoretical_infinite_sample_coverage_pdf(
        alpha=plot_params_1['alpha'], 
        calib_size=plot_params_1['calib_size']
    )
    return


@app.cell
def __():
    plot_params_2 = {
        'alpha': 0.1,
        'calib_sizes': [10, 100, 1_000, 10_000],
    }
    return (plot_params_2,)


@app.cell(hide_code=True)
def __(alt, np, pd, theoretical_infinite_sample_coverage_pdf):
    def plot_theoretical_infinite_sample_coverage_pdf_by_calib_size(alpha: float, calib_sizes: list[int], xlims: tuple[float, float] | None = None, npoints: int = 1_000) -> alt.Chart:
        if xlims is None:
            covered_frac = np.linspace(0, 1, npoints)
        else:
            covered_frac = np.linspace(xlims[0], xlims[1], npoints)
        
        dfs: list[pd.DataFrame] = []
        for calib_size in calib_sizes:
            pdf_values = theoretical_infinite_sample_coverage_pdf(covered_frac, alpha=alpha, calib_size=calib_size)    
            df = pd.DataFrame({
                'covered_frac': covered_frac,
                'pdf': pdf_values,
                'calib_size': calib_size
            }) 
            dfs.append(df)
        df = pd.concat(dfs)
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X('covered_frac:Q', title='Coverage', axis=alt.Axis(format=".0%")),
            y=alt.Y('pdf:Q', title='Probability Density'),
            color=alt.Color('calib_size:N', legend=alt.Legend(title="Calibration Size", format=',')) 
        ).properties(
            title=[
                f'Theoretical Coverage Distribution for α = {alpha:.1%} ',
                'and Infinitely Many Validation Points'
            ]
        )

        vline = 1
        
        return chart
    return (plot_theoretical_infinite_sample_coverage_pdf_by_calib_size,)


@app.cell(hide_code=True)
def __(
    plot_params_2,
    plot_theoretical_infinite_sample_coverage_pdf_by_calib_size,
):
    plot_theoretical_infinite_sample_coverage_pdf_by_calib_size(
        alpha=plot_params_2['alpha'], 
        calib_sizes=plot_params_2['calib_sizes'],
        xlims=(0.8, 1)
    ).properties(
        width=400,
        height=300
    )
    return


@app.cell
def __(mo):
    mo.md(r"""## Putting it into Practice""")
    return


@app.cell
def __(np):
    def sample_from_true_distribution(sample_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        noise_std = 0.1
        X = rng.uniform(low=-5, high=5, size=sample_size)
        mean = 0.5 + 0.5 * X + 0.1 * X**2
        y = mean + (X**2)*noise_std*rng.normal(loc=0, scale=1, size=sample_size)
        X = X.reshape(-1,1)
        y = y.reshape(-1)
        return X, y
    return (sample_from_true_distribution,)


@app.cell
def __(LinearRegression, Pipeline, PolynomialFeatures, make_pipeline, np):
    def train_model(X: np.ndarray, y: np.ndarray, degree: int = 2) -> Pipeline:
        model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
        model.fit(X, y)
        return model
    return (train_model,)


@app.cell
def __(Pipeline, np):
    def conformal_calibrate_model(
        model: Pipeline,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        alpha: float
    ) -> np.ndarray:

        # Compute non-conformity scores on calibration dataset:
        y_pred = model.predict(X_calib)
        scores = np.abs(y_calib - y_pred)

        # Compute (1-alpha) quantile of scores:
        scores = np.sort(scores, axis=-1)
        calib_size = y_calib.size
        qhat_idxs = np.ceil((calib_size + 1) * (1 - alpha)).astype(int) - 1
        qhat = np.append(scores, [np.inf])[qhat_idxs]

        return qhat
    return (conformal_calibrate_model,)


@app.cell
def __(Pipeline, np, pd):
    def compute_prediction_interval_coverage(
        model: Pipeline,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        qhat: float,  
    ) -> pd.DataFrame:

        # Form prediction intervals:
        y_pred = model.predict(X_valid)
        lb = y_pred - qhat
        ub = y_pred + qhat

        # Check if observed value falls within interval:
        is_covered = (lb <= y_valid) & (y_valid <= ub)

        # Stack results into DataFrame:
        results = pd.DataFrame(
            {
                'X': X_valid.flatten(),
                'y_obs': y_valid,
                'y_pred': y_pred,
                'lb': lb,
                'ub': ub,
                'is_covered': is_covered
            }
        )

        return results
    return (compute_prediction_interval_coverage,)


@app.cell
def __(
    Pipeline,
    compute_prediction_interval_coverage,
    conformal_calibrate_model,
    np,
    pd,
    sample_from_true_distribution,
):
    def split_conformal_prediction_with_trained_model(
        model: Pipeline,
        calib_size: int,
        valid_size: int,
        alpha: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, pd.DataFrame]:

        # Conformally calibrate trained model:
        X_calib, y_calib = sample_from_true_distribution(
            sample_size=calib_size,
            rng=rng,
        )
        qhat = conformal_calibrate_model(
            model=model, 
            X_calib=X_calib,
            y_calib=y_calib,
            alpha=alpha
        )

        # Check coverage of interval with validation dataset:
        X_valid, y_valid = sample_from_true_distribution(
            sample_size=valid_size,
            rng=rng,
        )
        df_coverage = compute_prediction_interval_coverage(
            model=model,
            X_valid=X_valid,
            y_valid=y_valid,
            qhat=qhat,
        )

        return qhat, df_coverage
    return (split_conformal_prediction_with_trained_model,)


@app.cell
def __(
    Pipeline,
    np,
    pd,
    sample_from_true_distribution,
    split_conformal_prediction_with_trained_model,
    train_model,
):
    def split_conformal_prediction_without_trained_model(
        train_size: int,
        calib_size: int,
        valid_size: int,
        alpha: float,
        rng: np.random.Generator,
    ) -> tuple[Pipeline, np.ndarray, pd.DataFrame]:


        # Train model with training dataset:
        X_train, y_train = sample_from_true_distribution(
            sample_size=train_size,
            rng=rng,
        )
        model = train_model(
            X=X_train,
            y=y_train,
        )

        qhat, df_coverage = split_conformal_prediction_with_trained_model(
            model=model,
            calib_size=calib_size,
            valid_size=valid_size,
            alpha=alpha,
            rng=rng,
        )

        return model, qhat, df_coverage
    return (split_conformal_prediction_without_trained_model,)


@app.cell
def __():
    train_size_1 = 10
    calib_size_1 = 100
    valid_size_1 = 100
    alpha_1 = 0.25
    seed_1 = 55
    return alpha_1, calib_size_1, seed_1, train_size_1, valid_size_1


@app.cell
def __(
    alpha_1,
    calib_size_1,
    np,
    seed_1,
    split_conformal_prediction_without_trained_model,
    train_size_1,
    valid_size_1,
):
    rng_1 = np.random.default_rng(seed_1)
    model_1, qhat_1, df_coverage_1 = split_conformal_prediction_without_trained_model(
        train_size=train_size_1,
        calib_size=calib_size_1,
        valid_size=valid_size_1,
        alpha=alpha_1,
        rng=rng_1
    )
    return df_coverage_1, model_1, qhat_1, rng_1


@app.cell(hide_code=True)
def __(Pipeline, alt, np, pd):
    def plot_model_with_prediction_interval(
        model: Pipeline,
        X: np.ndarray,
        qhat: float,
        opacity: float = 0.3,
        interval_color: str='lightgray',
        line_color: str='black',
        line_width: float = 3.,
    ) -> alt.Chart:
        y_pred = model.predict(X.reshape(-1,1))
        df = pd.DataFrame(
            {
                'X': X.flatten(), 
                'y': y_pred,
                'lb': y_pred - qhat,
                'ub': y_pred + qhat
            }
        )
        base = alt.Chart(df).encode(
            x=alt.X('X:Q', title='X'),
        )
        line = base.mark_line(color=line_color, strokeWidth=line_width).encode(
            y=alt.Y('y:Q', title='y')
        )
        interval = base.mark_area(opacity=opacity).encode(
            y='lb:Q',
            y2='ub:Q',
            color=alt.value(interval_color)
        )
        # So line is drawn on-top of interval:
        plot = interval + line
        return plot
    return (plot_model_with_prediction_interval,)


@app.cell(hide_code=True)
def __(alt, np, pd):
    def plot_validation_points(
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        is_covered: np.ndarray,
        marker_size: int = 60,
        covered_color: str='#009E73',
        uncovered_color: str='#D55E00',
    ) -> alt.Chart:
        # Encode True/False as 'Covered'/'Uncovered':
        is_covered = np.where(is_covered, 'Covered', 'Uncovered')
        df = pd.DataFrame(
            {
                'X': X_valid.flatten(), 
                'y': y_valid,
                'is_covered': is_covered
            }
        )
        points = alt.Chart(df).mark_point(size=marker_size).encode(
            x='X:Q',
            y='y:Q',
            color=alt.Color(
                'is_covered:N',
                scale=alt.Scale(
                    # Labels of labeled/unlabeled points in legend:
                    domain=['Covered', 'Uncovered'],  
                    # Colors of labeled/unlabeled points:
                    range=[covered_color, uncovered_color]  
                ),
                legend=alt.Legend(title='Validation Points')
            ),
            tooltip=[
                alt.Tooltip('X:Q', title='X'),
                alt.Tooltip('y:Q', title='y')
            ]
        )
        return points
    return (plot_validation_points,)


@app.cell(hide_code=True)
def __(
    alpha_1,
    alt,
    calib_size_1,
    df_coverage_1,
    model_1,
    np,
    pd,
    plot_model_with_prediction_interval,
    plot_theoretical_finite_sample_coverage_pdf,
    plot_validation_points,
    qhat_1,
    valid_size_1,
):
    # First panel = Model + Conformal Prediction Interval:
    X_plot_1 = np.linspace(df_coverage_1['X'].min(), df_coverage_1['X'].max(), num=100)
    plot_1a = (
        plot_model_with_prediction_interval(
        model=model_1,
        X=X_plot_1,
        qhat=qhat_1
        ) +
        plot_validation_points(
            X_valid=df_coverage_1['X'].to_numpy(),
            y_valid=df_coverage_1['y_obs'].to_numpy(),
            is_covered=df_coverage_1['is_covered'].to_numpy()
        )
    )
    num_covered_1 = df_coverage_1['is_covered'].sum()
    plot_1a = plot_1a.properties(
        title=[ 
            f"Model Predictions with {1-alpha_1:.1%} Conformal Interval",
            (
                f'({calib_size_1:,} Calibration Points, '
                f"{num_covered_1:,}/{valid_size_1:,} Validation Points Covered)"
            )
        ]
    )

    # Second panel = Theoretical Coverage Dist + Line Indicating Observed Coverage:
    vertline_1 = alt.Chart(
        pd.DataFrame(
            {
                'x': [num_covered_1],
            }
    )
    ).mark_rule(
        strokeWidth=1,
        color='#D55E00',
        strokeDash=[6, 6] # Dashed line
    ).encode(
        x='x:Q',
    )
    plot_1b = plot_theoretical_finite_sample_coverage_pdf(
        alpha=alpha_1,
        calib_size=calib_size_1,
        valid_size=valid_size_1
    ) + vertline_1

    # Plot panels side-by-side:
    plot_1a | plot_1b
    return X_plot_1, num_covered_1, plot_1a, plot_1b, vertline_1


@app.cell
def __(mo):
    mo.md(r"""Experiment""")
    return


@app.cell
def __(
    mo,
    np,
    pd,
    sample_from_true_distribution,
    split_conformal_prediction_with_trained_model,
    split_conformal_prediction_without_trained_model,
    train_model,
):
    def sample_split_conformal_coverage_distribution(
        num_samples: int,
        train_size: int,
        calib_size: int,
        valid_size: int,
        alpha: float,
        training_conditional: bool,
        rng: np.random.Generator
    ) -> pd.DataFrame:

        if training_conditional:
            X_train, y_train = sample_from_true_distribution(
                sample_size=train_size,
                rng=rng
            )
            model = train_model(X=X_train, y=y_train)

        results: list[dict[str, int | float]] = []
        for sample_idx in mo.status.progress_bar(
            range(num_samples), 
            total=num_samples,
            title='Sampling Coverage Distribution...'
        ):
            if training_conditional:
                qhat, df_coverage = split_conformal_prediction_with_trained_model(
                    model=model,
                    calib_size=calib_size,
                    valid_size=valid_size,
                    alpha=alpha,
                    rng=rng
                )
            else:
                _, qhat, df_coverage = split_conformal_prediction_without_trained_model(
                    train_size=train_size,
                    calib_size=calib_size,
                    valid_size=valid_size,
                    alpha=alpha,
                    rng=rng
                )
            result_i = {
                'sample_idx': sample_idx,
                'qhat': qhat,
                'num_covered': df_coverage['is_covered'].sum(),        
            }
            results.append(result_i)

        return pd.DataFrame.from_records(results)
    return (sample_split_conformal_coverage_distribution,)


@app.cell
def __():
    num_coverage_samples_2 = 1_000
    train_size_2 = 100
    calib_size_2 = 100
    valid_size_2 = 10
    alpha_2 = 0.1
    seed_2: int = 41
    training_conditional_2: bool = True
    return (
        alpha_2,
        calib_size_2,
        num_coverage_samples_2,
        seed_2,
        train_size_2,
        training_conditional_2,
        valid_size_2,
    )


@app.cell
def __(
    alpha_2,
    calib_size_2,
    np,
    num_coverage_samples_2,
    sample_split_conformal_coverage_distribution,
    seed_2,
    train_size_2,
    training_conditional_2,
    valid_size_2,
):
    rng_2 = np.random.default_rng(seed_2)
    df_samples_2 = sample_split_conformal_coverage_distribution(
        num_samples=num_coverage_samples_2,
        train_size=train_size_2,
        calib_size=calib_size_2,
        valid_size=valid_size_2,
        alpha=alpha_2,
        training_conditional=training_conditional_2,
        rng=rng_2
    )
    return df_samples_2, rng_2


@app.cell
def __(df_samples_2):
    df_samples_2
    return


@app.cell(hide_code=True)
def __(
    alpha_2,
    alt,
    calib_size_2,
    df_samples_2,
    np,
    num_coverage_samples_2,
    pd,
    theoretical_finite_sample_coverage_pdf,
    valid_size_2,
):
    def plot_observed_coverage_against_theory(
        df_obs: pd.DataFrame,
        alpha: float,
        calib_size: int,
        valid_size: int,
        opacity: float = 0.5,
        observed_color: str='#0072B2',
        theory_color: str='#D55E00',
    ):
        # Compute observed coverage counts:
        obs_pmf: np.ndarray = df_obs.groupby('num_covered').size().reset_index(name='pmf')
        obs_pmf['pmf'] = obs_pmf['pmf'] / obs_pmf['pmf'].sum()
        min_obs_covered = obs_pmf['num_covered'].min()
        obs_pmf['group'] = 'Observed'

        # Compute theoretical coverage counts:
        kmin = obs_pmf['num_covered'].min()
        kmax = obs_pmf['num_covered'].max()
        k_values = np.arange(kmin, kmax+1)
        probs = [
            theoretical_finite_sample_coverage_pdf(k, alpha=alpha, valid_size=valid_size, calib_size=calib_size) 
            for k in k_values
        ]   
        theo_pmf = pd.DataFrame({
            'num_covered': k_values,
            'pmf': probs
        })
        theo_pmf['group'] = 'Theory'

        df = pd.concat([obs_pmf, theo_pmf])
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('num_covered:O', title='Number of Covered Validation Samples', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('pmf:Q', title='Probability Mass'),
            color=alt.Color(
                'group:N',
                scale=alt.Scale(
                    domain=['Observed', 'Theory'],
                    range=[observed_color, theory_color]
                )
            ),
            xOffset=alt.XOffset("group:N"),
            tooltip=[
                alt.Tooltip('num_covered', title='Number of Covered Samples'), 
                alt.Tooltip('pmf', title='Probability Mass'),
                alt.Tooltip('group', title='Group'),
            ]
        )

        return chart

    plot_observed_coverage_against_theory(
        df_obs=df_samples_2,
        alpha=alpha_2,
        calib_size=calib_size_2,
        valid_size=valid_size_2,
    ).properties(
        width=600,
        height=400,    
        title=[ 
            f"Observed v Theoretical Coverage Distribution",
            f"(Confidence={1-alpha_2:.1%}, Calibration Samples={calib_size_2:,}, ",
            f" Validation Samples={valid_size_2:,}, Coverage Samples={num_coverage_samples_2:,})"

        ]
    )
    return (plot_observed_coverage_against_theory,)


@app.cell(hide_code=True)
def __(alt, df_samples_2):
    alt.Chart(df_samples_2).mark_point(
        color='#D55E00'
    ).encode(
        x=alt.X(
            'num_covered:Q', 
            title='Number of Covered Validation Samples',
            scale=alt.Scale(
                domain=[
                    df_samples_2['num_covered'].min(),
                    df_samples_2['num_covered'].max(),
                ]
            )
        ),
        y=alt.Y(
            'qhat:Q', 
            title='Prediction Interval Half-Width',
            scale=alt.Scale(
                domain=[
                    df_samples_2['qhat'].min(),
                    df_samples_2['qhat'].max(),
                ]
            )
        ),
        tooltip=[
            alt.Tooltip('num_covered', title='Number of Covered Samples'), 
            alt.Tooltip('qhat', title='Interval Half-Width')
        ]
    ).properties(
        title='Coverage vs Prediction Interval Size',
        width=400,
        height=300
    )
    return


if __name__ == "__main__":
    app.run()
