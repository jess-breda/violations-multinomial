import dynamax
from functools import partial

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import vmap
from jax.nn import one_hot
import optax
from sklearn.preprocessing import StandardScaler

from multiglm.utils.dynamax_utils import *

from dynamax.hidden_markov_model import LogisticRegressionHMM, CategoricalRegressionHMM


num_states = 3
input_dim = 2
glmhmm = LogisticRegressionHMM(
    num_states=num_states, input_dim=input_dim, transition_matrix_stickiness=5
)
params, _ = glmhmm.initialize(key=jr.PRNGKey(19))

print_binary_hmm_params(params)

num_timesteps = 300
key = jr.PRNGKey(19)
key1, key2 = jr.split(key)
inputs = jr.uniform(key2, shape=(num_timesteps, input_dim))

key = jr.PRNGKey(19)
key1, key2 = jr.split(key)
true_states, emissions = glmhmm.sample(
    params,
    key1,
    num_timesteps=num_timesteps,
    inputs=inputs,  # for regression problem must specify inputs!! is there a proper way to do this? should these
)
