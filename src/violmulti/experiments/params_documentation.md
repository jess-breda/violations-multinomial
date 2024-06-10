# Experiment Params Documentation 

## All Experiments

These are parameters that are used by all experiment classes. If optional, default is labeled.

`animals`: list of strings, default = None
    if None, all animals in dataset are loaded and fit

`data_type`: for data loader

`sigmas` : list of float64
    sigma terms to sweep over for MAP penalty for binary or multi-class logistic regression (if None or 0, no penalty)

`random_state`: int, default = 47

`test_size`: float64, default = 0.2

`null_mode` : str, default = 
    binary or multi

`eval_train` : bool, default = False # TODO remove?

`null_model` : optional, if exits 'binary' or 'multi'

`min_training_stage` : int
    used for taus df and filtering of df

`tau_columns` : list 
    can also be none? with sigma sweep

`model_config`: sub dictionary with model params for model you want to fit represented as a model name key and sub-dictionary of params

    `model_1_name` : sub dictionary 

        `model_class`:

        `model_type`:

        `design_matrix_generator` : class

        `design_matrix_generator_args`: sub dict

            `interaction_pairs`:

            `XXX`: ?
        
        `filter_implementation`: sub dict, default =
            
            `column_name_1` : int, -1, 0 or 1

            `column_name_2` : int, -1, 0 or 1

            if None or {} then no filter params

        `lr_only_eval` : bool, default = False





## Sigma Sweep Experiment



## Tau Sweep Experiment

`taus`:

`tau_sweep`: sub dictionary 
```
    "tau_sweep": {
        "prev_violation": True,
    },
```

## Compare Models

`tau_columns`

## Compare Binary Multi

=
