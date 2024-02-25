import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def fit_linear_model(df, predictor, features, whiten=False, plot=True):
    """ """
    if whiten:
        # Standardize (whiten) the features
        X_unscaled = df[features]  # Features
        scaler = StandardScaler()
        X = scaler.fit_transform(X_unscaled)
    else:
        X = df[features]  # Features

    y = df[predictor]
    model = LinearRegression()
    model.fit(X, y)

    # Predict the target variable using the trained model
    predictions = model.predict(X)
    r_squared = model.score(X, y)

    # Assuming 'model' is your trained LinearRegression model
    weights = model.coef_
    bias = model.intercept_
    regressor_names = df[features].columns.tolist()

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_weights(
            ax, regressor_names, weights, title=f"{predictor}, $R^2$ = {r_squared:.2f}"
        )
        return model, predictions, r_squared
    else:
        return model, predictions, r_squared, regressor_names, weights


def plot_weights(ax, regressor_names, weights, title, **kwargs):
    """ """
    ax.bar(regressor_names, weights, **kwargs)
    ax.axhline(y=0, color="k")

    ax.set(xlabel="", ylabel="Weight", title=title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # plt.xticks(rotation=90)  # Rotating the x-axis labels if needed


def plot_predictions(ax, predictor, predictions, title):
    """ """
    ax.scatter(predictor, predictions)
    ax.axline([0, 0], [1, 1], color="k")

    data_range = max(predictor.max(), predictions.max()) * 1.2
    ax.set(
        xlim=(0, data_range),
        ylim=(0, data_range),
        xlabel="Actual",
        ylabel="Predicted",
        title=title,
    )


def univariate_linear_regression(df, x, y, print_results=True):
    """
    Performs univariate linear regression on given data.

    params
    -------
    df: pd.DataFrame
        Dataframe containing the data.
    x: str
        Name of the feature to be used as predictor.
    y: str
        Name of the feature to be used as target.

    Returns:
    - results (RegressionResults): Fitted regression model results.
    """

    X = df[x].values.reshape(-1, 1)
    y = df[y].values.reshape(-1, 1)
    X = sm.add_constant(X)

    model = sm.OLS(y, X)
    results = model.fit()

    if print_results:
        print(results.summary())

    return results


def plot_univariate_linear_regression(results, ax=None, **kwargs):
    """
    Plots the linear regression model and data points.

    Parameters:
    - results (RegressionResults): Fitted regression model results.
    - ax (matplotlib axis, optional): Axis on which to plot.
    - kwargs (dict): Additional settings for plot axis.

    Returns:
    - ax (matplotlib axis): Axis containing the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    slope = results.params[1]
    intercept = results.params[0]

    # Get x and y data from the results
    x_data = results.model.exog[:, 1]
    y_data = results.model.endog

    # Determine the range for the line model
    x_line = np.linspace(min(x_data) * 0.5, max(x_data) * 1.5, 100)
    y_line = slope * x_line + intercept

    ax.scatter(x_data, y_data, color="black", label="Data")
    ax.plot(x_line, y_line, color="salmon", label="Regression Line")

    # Set axis limits, labels, title etc. based on kwargs
    ax.set(**kwargs)

    # Add text to upper right corner
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.text(xlim[1] * 0.75, ylim[1] * 0.75, f"$R^2$ = {results.rsquared_adj:.2f}")

    return ax
