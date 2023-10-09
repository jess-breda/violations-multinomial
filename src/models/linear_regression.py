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


def plot_weights(ax, regressor_names, weights, title):
    """ """
    ax.bar(regressor_names, weights)
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
