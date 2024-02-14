# Dataset Thresholding
> Written by Jess Breda 2024-02-14

## Overview 
These files are related to the notebook `create_trained_threshold_df.ipynb` where I explored the cleaned version of the new (2024 release) dataset to determine when animals are trained and which sessions to use for modeling. This work generated the `all_animals_trained_threshold.csv` table.

To do this, I started by taking rolling averages of hit and violation rates with various window sizes (7, 14, 21 and 28) and thresholds (hit: 70, 72.5 and 75%, violation: 50%).

I determined the start of the "trained" window as the last crossing from low to high through the 72.5% hit threshold for a 21 day rolling average. 

The end of the trained window was either the animals final sessions (coded as violation_threshold = nan), or for some hand selected animals who have a peak in violation rates and a crash in hit rates, the last crossing from low to high through the 50% threshold for a 7 day rolling average.

The notebook goes into additional details as to why these window sizes and thresholds were picked.

## Documentation
The related files can be described as follows:

> `threshold_data_dict.pkl`: dictionary
* **hit:**

    * **rolling_mean_window_df**: rolling mean of session hit rate for all window sizes tested and concatenated together

    * **crossing_thresholds_df**: data frame with sessions for different cross points (min, median, max), with final window size and hit rate at 70%

    * **window_size**: final window size used for rolling mean (21)

    * **threshold**: final hit rate used for threshold (72.5%)

* **violation:**

    * **rolling_mean_window_df**: rolling mean session violation rate for all window sizes tested and concatenated together

    * **crossing_thresholds_df**: data frame with sessions for different cross points (min, median, max), with final window size and violation rate at 50%

    * **window_size**: final window size used for rolling mean (7)

    * **threshold**: final hit rate used for threshold (50%)

---
> `hit_21_viol_7_rolling_means.csv` : pd.DataFrame

* hit and violation **rolling_mean_window_df** filtered for final selected window size (21 and 7, respectively) and merged together with animal_id, session as row index.
---

> `threshold_sessions.csv` : pd.DataFrame

* DataFrame with columns: animal_id, hit_threshold (start session), violation_threshold (end session) with animal_id as row index. Used to filter cleaned data to create `all_animals_trained_threshold.csv`
---
> `trained_threshold_df_stats.csv`: pd.DataFrame

* DataFrame with summary statistics of `all_animals_trained_threshold.csv` with animal_id as row index. Contains stats on: number of trials, number of sessions, avg hit and avg violation rate.

## Related Code

See `src/data/dataset_thresholder.py`

