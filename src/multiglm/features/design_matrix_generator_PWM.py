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
    def __init__(self, df, config):

        super().__init__(df, config)
        self.X["choice"] = df.choice  # FOR DEBUG INIT
        self.run_init_tests()

    def run_init_tests(self):

        assert (
            len(self.df["animal_id"].unique()) == 1
        ), "More than 1 animal in dataframe!"

    def create(self):
        X = super().create()

        return X


## METHODS

# prev violation mask
# prev sound average
# prev choice
# prev correct

# binary and multi label maps?
