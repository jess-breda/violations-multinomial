import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from violmulti.visualizations.model_visualizer import ModelVisualizer

"Written 2025-04-14 for thesis work, cleaned up and specified version of pyschometrics.py"

def sigmoid(z):
    """Compute the logistic function."""
    return 1 / (1 + np.exp(-z))


class SimulatePrevViolPyschometric:
    """
    Class to simulate and plot psychometric curves from R-L weights of a multi-class GLM.
    
    The simulation uses a 1D grid for delta (s_a - s_b) ranging from -4 to 4.
    Features must match:
        - "bias"
        - "s_a"
        - "sa_x_prev_viol"
        - "s_b"
        - "sb_x_prev_viol"
    where the interaction terms for the violation history (v_x) are represented by
    "sa_x_prev_viol" and "sb_x_prev_viol". The weights are computed as (R - L).
    
    You should instantiate this class by passing your experiment (which contains the model fit)
    and the model name.
    """
    
    def __init__(self, experiment, model_name, features_of_interest=None):
        """
        Initialize with an experiment object and model_name.
        
        Parameters:
        -----------
        experiment : object
            Your experiment data (to be passed to ModelVisualizer).
        model_name : str
            The name of the model (e.g., "prev_viol_binary_itrx") to filter the weights.
        features_of_interest : list of str, optional
            List of feature names to use. Defaults to:
            ["bias", "s_a", "sa_x_prev_viol", "s_b", "sb_x_prev_viol"].
        """
        if features_of_interest is None:
            features_of_interest = ["bias", "s_a", "sa_x_prev_viol", "s_b", "sb_x_prev_viol"]
        self.features_of_interest = features_of_interest
        self.model_name = model_name
        self.experiment = experiment
        
        # Use ModelVisualizer from your package to extract the fitted model data.
        # (Make sure ModelVisualizer is imported from your package.)
        self.mv = ModelVisualizer(experiment)
        flattened_df = self.mv.unpack_features_and_weights(self.mv.find_best_fit())
        
        # Filter flattened_df to only include desired features and weights from L and R classes 
        # and for the specific model.
        self.flattened_df = flattened_df.query(
            "feature in @features_of_interest and weight_class in ['L', 'R'] and model_name == @model_name"
        ).copy()
        
        # Compute the difference between R and L weights for each animal and feature.
        self.delta_weights_df = self._compute_delta_weights()
    
    def _compute_delta_weights(self):
        """
        Compute the delta weights (R - L) for each animal and each feature.
        
        Returns:
        --------
        delta_weights_df : pd.DataFrame
            DataFrame with columns: animal_id, feature, weight (R - L), and weight_class = "R-L".
        """
        data_list = []
        for (aid, feat), group in self.flattened_df.groupby(["animal_id", "feature"]):
            # Make sure both classes ('L' and 'R') are present.
            if set(group['weight_class']) >= {"L", "R"}:
                L_w = group.query("weight_class == 'L'")["weight"].values[0]
                R_w = group.query("weight_class == 'R'")["weight"].values[0]
                delta_w = R_w - L_w
                data_list.append({
                    "animal_id": aid,
                    "feature": feat,
                    "weight": delta_w,
                    "weight_class": "R-L"
                })
        return pd.DataFrame(data_list)
    
    def get_animal_ids(self):
        """
        Return the unique animal IDs from the computed delta weights.
        
        Returns:
        --------
        np.array
            Array of unique animal IDs.
        """
        return self.delta_weights_df["animal_id"].unique()
    
    def simulate_psych(self, animal_id, filter_val=0, delta_range=(-2, 2), num_points=500):
        """
        Simulate the psychometric function for a given animal and filter value.
        
        The simulation assumes a 1D grid where s_a = delta/2 and s_b = -delta/2 so that (s_a - s_b) = delta.
        The decision variable is computed as:
        
            z = ((w_sa + w_sa_x * v_x) * s_a) + ((w_sb + w_sb_x * v_x) * s_b) + bias
        
        and p_right = sigmoid(z).
        
        Parameters:
        -----------
        animal_id : str
            The ID of the animal to simulate.
        filter_val : int, optional
            The filter value (v_x); typically 0 or 1.
        delta_range : tuple, optional
            The range of delta (s_a - s_b) values. Default is (-4, 4).
        num_points : int, optional
            Number of points in the delta grid.
            
        Returns:
        --------
        sim_df : pd.DataFrame
            DataFrame with columns: delta, p_right, v_x, and animal_id.
        """
        # Extract weights for the given animal.
        df_animal = self.delta_weights_df[self.delta_weights_df["animal_id"] == animal_id]
        try:
            w_sa   = df_animal.loc[df_animal["feature"] == "s_a", "weight"].values[0]
            w_sb   = df_animal.loc[df_animal["feature"] == "s_b", "weight"].values[0]
            w_sa_x = df_animal.loc[df_animal["feature"] == "sa_x_prev_viol", "weight"].values[0]
            w_sb_x = df_animal.loc[df_animal["feature"] == "sb_x_prev_viol", "weight"].values[0]
            bias   = df_animal.loc[df_animal["feature"] == "bias", "weight"].values[0]
        except IndexError as e:
            raise ValueError(f"Missing weight features for animal {animal_id}: {e}")
        
        # Create a 1D grid for delta.
        delta_array = np.linspace(delta_range[0], delta_range[1], num_points)
        # Define s_a and s_b such that s_a - s_b = delta.
        s_a = delta_array / 2
        s_b = -delta_array / 2
        v_x = filter_val
        # Compute decision variable z.
        z = ((w_sa + w_sa_x * v_x) * s_a +
             (w_sb + w_sb_x * v_x) * s_b +
             bias)
        # Compute probability using sigmoid.
        p_right = sigmoid(z)
        sim_df = pd.DataFrame({
            "delta": delta_array,
            "p_right": p_right,
            "v_x": v_x,
            "animal_id": animal_id
        })
        return sim_df
    
    def plot_animal_psychometric(self, animal_id, detla_range, ax=None):
        """
        Plot the psychometric curves for an individual animal for both filter values (0 and 1).
        
        Parameters:
        -----------
        animal_id : str
            The animal identifier.
        ax : matplotlib.axes.Axes, optional
            If provided, the curves are plotted on this axis.
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axis containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Simulate and plot for both filter values.
        for f_val in [0, 1]:
            sim_df = self.simulate_psych(animal_id, delta_range=detla_range, filter_val=f_val)
            sns.lineplot(data=sim_df, x="delta", y="p_right", ax=ax,
                         label=f"v_x = {f_val}")
        
        ax.set_xlabel("$s_a - s_b$")
        ax.set_ylabel("P(Right)")
        ax.set_title(f"Psychometric Curve for Animal ID: {animal_id}")
        ax.set_ylim(0, 1)
        ax.legend(title="Prev Viol Filter")
        plt.show()
        return ax
    
    def plot_summary(self, animal_filter=None,delta_range=(-2, 2), title=None, **kwargs):
        """
        Plot a summary psychometric curve across all animals. For each animal and each filter value,
        the simulation is performed. The plot uses the filter value as the hue and groups by animal_id.
        
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axis containing the summary plot.
        """
        simulation_results = []
        for animal_id in self.get_animal_ids():
            for f_val in [0, 1]:
                sim_df = self.simulate_psych(animal_id, delta_range=delta_range, filter_val=f_val)
                simulation_results.append(sim_df)
        all_animal_df = pd.concat(simulation_results, ignore_index=True)
        self.all_animal_df = all_animal_df
        summary_df = all_animal_df.groupby(["animal_id","delta", "v_x"]).mean().reset_index()

        if animal_filter:
            summary_df = summary_df.query("animal_id in @animal_filter")
        
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.despine()
        sns.lineplot(
            data=summary_df,
            x="delta",
            y="p_right",
            hue="v_x",
            ax=ax,
            errorbar="se",
            style="v_x",
            **kwargs
        )


        ax.set_xlabel("$s_a - s_b$")
        ax.set_ylabel("P(Right)")
        ax.set_title(title if title else "Summary Psychometric Curves Across Animals")
        ax.set_ylim(-.1, 1.1)
        handles, labels = ax.get_legend_handles_labels()
        labels = ["False" if label == "0" else "True" for label in labels]
        ax.legend(handles, labels, title="Prev Violation", frameon=False)

        return fig, ax
    def extract_slope_and_lapse(self, delta_range=(-2, 2), num_points=500):
        """
        Extract the slope at delta=0 and the lapse (upper and lower) for each animal
        and each condition (v_x = 0 or 1).
        
        The slope is derived from the derivative of the logistic at delta=0.
        Specifically, if z = M * delta + bias, then:
            M = 0.5 * [(w_sa - w_sb) + v_x * (w_sa_x - w_sb_x)]
        and slope at delta=0 is M * logistic(bias) * (1 - logistic(bias)).
        
        The upper lapse is defined as (1 - p_right) at the maximum delta in delta_range.
        The lower lapse is defined as p_right at the minimum delta in delta_range.
        
        Parameters:
        -----------
        delta_range : tuple, optional
            The (min, max) range of delta for the simulation.
        num_points : int, optional
            Number of delta points for simulating p_right.
        
        Returns:
        --------
        pd.DataFrame
            A DataFrame with columns:
             - animal_id
             - v_x
             - slope
             - upper_lapse
             - lower_lapse
        """
        results = []
        
        for animal_id in self.get_animal_ids():
            # Pull out relevant R-L weights for this animal
            df_animal = self.delta_weights_df.query("animal_id == @animal_id")
            try:
                w_sa   = df_animal.loc[df_animal["feature"] == "s_a", "weight"].values[0]
                w_sb   = df_animal.loc[df_animal["feature"] == "s_b", "weight"].values[0]
                w_sa_x = df_animal.loc[df_animal["feature"] == "sa_x_prev_viol", "weight"].values[0]
                w_sb_x = df_animal.loc[df_animal["feature"] == "sb_x_prev_viol", "weight"].values[0]
                bias   = df_animal.loc[df_animal["feature"] == "bias", "weight"].values[0]
            except IndexError as e:
                # If for some reason we can't find the necessary features, skip or raise an error
                raise ValueError(f"Missing weight features for animal {animal_id}: {e}")
            
            for v_x in [0, 1]:
                # Derive M = derivative wrt delta
                # z = M*delta + bias
                M = 0.5 * ((w_sa - w_sb) + v_x*(w_sa_x - w_sb_x))
                
                # Probability at delta=0
                p0 = sigmoid(bias)  # logistic(bias)
                # Slope at delta=0 = M * p0*(1-p0)
                slope_0 = M * p0 * (1 - p0)
                
                # Now simulate the full function and find the asymptotic behavior for lapse
                sim_df = self.simulate_psych(animal_id, filter_val=v_x,
                                             delta_range=delta_range,
                                             num_points=num_points)
                # p_right at largest delta
                max_delta_idx = sim_df["delta"].idxmax()
                # p_right at smallest delta
                min_delta_idx = sim_df["delta"].idxmin()
                
                p_high = sim_df.loc[max_delta_idx, "p_right"]  # near the high end
                p_low  = sim_df.loc[min_delta_idx, "p_right"]  # near the low end
                
                # Common definitions of "lapse":
                upper_lapse = 1 - p_high
                lower_lapse = p_low
                
                results.append({
                    "animal_id": animal_id,
                    "v_x": v_x,
                    "slope": slope_0,
                    "upper_lapse": upper_lapse,
                    "lower_lapse": lower_lapse
                })
        
        return pd.DataFrame(results)