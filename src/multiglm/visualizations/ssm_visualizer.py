import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List, Union
from ssm import *


# Transition Matrix
def plot_transition_matrix(
    log_trans_mat: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: Optional[str] = "bone",
    title: Optional[str] = "Transition Matrix",
) -> None:
    """
    Plots a heatmap of the transition matrix with annotations.

    Parameters
    ----------
    log_trans_mat : np.ndarray
        The transition matrix to be plotted in log space
    ax : matplotlib.axes.Axes, (default=None)
        The axis to plot to.
    cmap : str (default='bone')
        The colormap to be used for the heatmap
    title : str (default='Transition Matrix')
        The title of the plot
    """
    # Convert log transition matrix to transition matrix
    trans_mat = np.exp(log_trans_mat)
    if len(trans_mat.shape) > 2:
        trans_mat = trans_mat[0]  # drop padding dimension

    # Plot the heatmap
    sns.heatmap(
        trans_mat,
        vmin=-0.8,
        vmax=1,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar=False,
    )

    ax.set(
        ylabel="state t",
        xlabel="state t+1",
        title=title,
    )

    return None


# State Posterior
#! TODO improve this code to work with various shapes over sessions single session, etc
def plot_state_posterior(
    posterior: np.ndarray,
    ax: Optional[plt.Axes] = None,
    sample_window: Tuple[int, int] = (0, None),
    title=None,
    **kwargs,
) -> None:
    """
    Plots the posterior state probabilities over time.

    Parameters
    ----------
    posterior : np.ndarray
        The array containing posterior state probabilities.
        Shape should be (num_timesteps, num_states).
    session_id : int
        The session to plot posterior for
    ax : matplotlib.axes.Axes, optional (default=None)
        The axis to plot on. If None, a new figure and axis are created.
    sample_window : Tuple[int, int], optional (default=(0, 150))
        The range of samples to plot (start, end).
    **kwargs : dict
        Additional keyword arguments for the plot function.

    Returns
    -------
    None
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    num_timesteps = posterior.shape[0]
    num_states = posterior.shape[1]
    cmap = sns.color_palette("husl", num_states)

    for i in range(num_states):
        ax.plot(posterior[:, i], color=cmap[i], label=f"State {i}", **kwargs)

    ax.set(
        xlabel="Trial",
        ylabel="P(state)",
        title="Smoothed Posterior Probabilities" if title is None else title,
        xlim=sample_window,
    )

    ax.legend(bbox_to_anchor=(1.0, 1), loc="upper left")


# Weights
def plot_bernoulli_weights_by_state(
    weights: np.ndarray,
    feature_names: List[str],
    ax: Optional[plt.Axes] = None,
    palette: Optional[str] = "husl",
    title: Optional[str] = "Bernoulli Weights",
    **kwargs,
) -> None:
    """
    Plots the state-dependent weights of a binary GLM.

    Parameters
    ----------
    weights : np.ndarray
        The array containing the weights.
        Shape should be (num_states, num_classes, num_features).
    feature_names : List[str]
        List of feature names corresponding to the features in the weights array.
    palette : str, optional (default="husl")
        Color palette to use for the plot.
    title : str, optional (default="Bernoulli Weights")
        Title of the plot.
    **kwargs : dict
        Additional keyword arguments for the plot function.

    Returns
    -------
    None
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    num_states, num_classes, num_features = weights.shape
    sns.set_palette(sns.color_palette(palette, num_states))

    ax.axhline(y=0, color="k")

    for state in range(num_states):
        # could iterate over classes here if you want
        ax.plot(
            feature_names,
            weights[state, 0, :],  # replace 0 with class_idx
            marker="o",
            label=f"State {state}",
            **kwargs,
        )

    # aesthetics
    ax.set(
        xlabel="Feature",
        ylabel="Weight",
        title=title,
    )
    ax.legend()
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45)

    return None


def plot_specified_class_weights_by_state(
    weights: np.ndarray,
    feature_names: List[str],
    class_idx: int,
    ax: Optional[plt.Axes] = None,
    palette: Optional[str] = "husl",
    title: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Plots the state-dependent weights of a binary GLM.

    Parameters
    ----------
    weights : np.ndarray
        The array containing the weights.
        Shape should be (num_states, num_classes, num_features).
    feature_names : List[str]
        List of feature names corresponding to the features in the weights array.
    class_idx : int
        Index of the class to plot weights for.
    palette : str, optional (default="husl")
        Color palette to use for the plot.
    title : str, optional (default="Bernoulli Weights")
        Title of the plot.
    **kwargs : dict
        Additional keyword arguments for the plot function.

    Returns
    -------
    None
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    num_states, num_classes, num_features = weights.shape
    sns.set_palette(sns.color_palette(palette, num_states))

    ax.axhline(y=0, color="k")

    for state in range(num_states):
        ax.plot(
            feature_names,
            weights[state, class_idx, :],
            marker="o",
            label=f"State {state}",
            **kwargs,
        )

    # aesthetics
    ax.set(
        xlabel="Feature",
        ylabel="Weight",
        title=title if title else f"Class {class_idx} Weights",
    )
    ax.legend()
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45)

    return None


## LP by iterations
def plot_log_probs_over_iters(
    fit_ll: List[float],
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = "LL by Iteration",
    true_ll: Optional[float] = None,
    **kwargs,
) -> None:
    """
    Plot log probs over iterations of EM algorithm

    Parameters
    ----------

    fit_ll : List[float]
            List of log probability values over iterations of EM algorithm
            returned by ssm.hmm.fit()
    ax : matplotlib.axes.Axes, (default=None)
            The axis to plot to.
    title : str, (default="LL by Iteration")
            The title of the plot
    true_ll : float, (default=None)
            The true log likelihood value if comparing to a known value
            that will be used to draw a horizontal line on the plot
    **kwargs : dict
            Additional keyword arguments for the plot function

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=80)

    ax.plot(fit_ll, **kwargs)
    ax.set(
        xlabel="EM Iteration",
        ylabel="Log Likelihood",
        title=title,
    )

    if true_ll is not None:
        ax.axhline(true_ll, color="black", linestyle="--", label="True LL")
        ax.legend()

    return None


# State Occupancy
def plot_state_occupancies(
    state_occupancies: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = None,
    **kwargs,
) -> None:
    """
    Plots the state occupancies as a bar chart.

    Parameters
    ----------
    state_occupancies : np.ndarray (n_states,)
        An array containing the occupancy fractions for each state.
    ax : matplotlib.axes.Axes, optional (default=None)
        The axis to plot on. If None, a new figure and axis are created.
    title : str, optional (default=None)
        The title of the plot. If None, "State Occupancies" is used.
    **kwargs : dict
        Additional keyword arguments for the sns.barplot function.

    Returns
    -------
    None
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    x = [f"state_{i}" for i in range(len(state_occupancies))]
    sns.barplot(x=x, y=state_occupancies, ax=ax, **kwargs)

    ax.set(
        ylim=(0, 1),
        title=title if title else "State Occupancies",
        xlabel=None,
        ylabel="Frac Occupancy",
    )

    return None


# Utils
# Get posterior probs


def get_posterior_state_probs(
    glmmhmm: "ssm.HMM",
    true_choices: Union[List[np.ndarray], List[List[np.ndarray]]],
    inputs: Union[List[np.ndarray], List[List[np.ndarray]]],
) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
    """
    Function description here.

    Parameters
    ----------
    glmmhmm :
        The SMM glm-hmm object that has been fit
    true_choices : ["num_sessions" ["num_trials_per_session"]] or ["num_trials"]
        True choices (y) can be in the form of a list of arrays for each
        session or a single concatenated array.
    inpts : ["num_sessions" ["num_trials_per_session"]] or ["num_trials"]
        Inputs (X) can be in the form of a list of arrays for each
        session or a single concatenated array.

    Returns
    -------
    posterior_probs :
        Posterior probabilities of the states for each session.
    """
    posterior_probs = []
    for sesssion_choices, sesssion_inputs in zip(true_choices, inputs):

        # expected_states returns
        # [posterior_state_probs, posterior_joint_porbs, nomrmalizer]
        # so we only need the first element of the returned list
        posterior_probs.append(
            glmmhmm.expected_states(data=sesssion_choices, input=sesssion_inputs)[0]
        )
    return posterior_probs


# Get state occupancy


def get_state_occupancies(
    glmmhmm: "ssm.HMM",
    true_choices: Union[List[np.ndarray], List[List[np.ndarray]]],
    inputs: Union[List[np.ndarray], List[List[np.ndarray]]],
) -> np.ndarray:
    """
    Computes the state occupancies from the given GLM-HMM model.

    Parameters
    ----------
    glmmhmm : ssm.HMM
        The GLM-HMM object that has been fit.
    true_choices : list of np.ndarray or list of list of np.ndarray
        True choices (y) can be in the form of a list of arrays for each
        session or a single concatenated array.
    inputs : list of np.ndarray or list of list of np.ndarray
        Inputs (X) can be in the form of a list of arrays for each
        session or a single concatenated array.

    Returns
    -------
    state_occupancies : np.ndarray (n_states,)
        An array containing the occupancy fractions for each state.
    """

    posterior_state_probs = get_posterior_state_probs(glmmhmm, true_choices, inputs)

    if len(posterior_state_probs[0]) > 1:  # remove session dimension if it exists
        posterior_state_probs = np.concatenate(posterior_state_probs)

    state_max_posterior = np.argmax(posterior_state_probs, axis=1)
    _, state_occupancies = np.unique(state_max_posterior, return_counts=True)
    state_occupancies = state_occupancies / np.sum(state_occupancies)

    return state_occupancies


# - get assigned state (state_max_posterior)
# - get fractional state occupancy
