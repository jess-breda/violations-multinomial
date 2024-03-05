"""
Child class of DesignMatrixGenerator for creating design matrices
specific to the PWM dataset.
"""

import pandas as pd
from pandas import Series
import numpy as np
from multiglm.features.design_matrix_generator import *


## CLASS
class DesignMatrixGeneratorPWM(DesignMatrixGenerator):
    def __init__(self, df, config, verbose=False):
        super().__init__(df, config, verbose)
        self.X["choice"] = df.choice  # FOR DEBUG INIT
        self.run_init_tests()

    def run_init_tests(self):

        assert (
            len(self.df["animal_id"].unique()) == 1
        ), "More than 1 animal in dataframe!"

    def create(self):
        X, y = super().create()

        return X, y


## METHODS

# prev violation mask
# prev sound average
# prev choice
# prev correct

# binary and multi label maps?
