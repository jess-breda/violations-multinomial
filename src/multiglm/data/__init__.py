ANIMAL_IDS = [
    "W051",
    "W060",
    "W061",
    "W062",
    "W065",
    "W066",
    "W074",
    "W075",
    "W078",
    "W080",
    "W081",
    "W082",
    "W083",
    "W084",
    "W088",
    "W089",
    "W094",
]


COLUMN_RENAME = {
    "rat_name": "animal_id",
    "session_date": "session_date",
    "session_counter": "session_file_counter",  # can have multiple in the same date
    "A1_dB": "s_a",
    "A2_dB": "s_b",
    "hit_history": "hit",
    "violation_history": "violation",
    "timeout_history": "trial_not_started",
    "A1_sigma": "s_a_sigma",  # not sure why this is here, dB mapping done in matlab
    "Rule": "rule",
    "ThisTrial": "correct_side",  # eventually: 1 = right, 0 = left
    "violation_iti": "violation_penalty_time",
    "error_iti": "error_penalty_time",
    "secondhit_delay": "delayed_reward_time",  # related to stg 3, doesn't tell you if animal used it though
    "PreStim_time": "pre_stim_time",
    "A1_time": "s_a_time",
    "Del_time": "delay_time",
    "A2_time": "s_b_time",
    "time_bet_aud2_gocue": "post_s_b_to_go_cue_time",
    "time_go_cue": "go_cue_time",
    "CP_duration": "fixation_time",
    "CenterLed_duration": "trial_start_wait_time",  # how much time elapses w/o activity until available trial is a "timeout"
    "Left_volume": "l_water_vol",
    "Right_volume": "r_water_vol",
    "Beta": "antibias_beta",  # higher = stronger antibias
    "RtProb": "antibias_right_prob",  # higher = more likely for a right trial to occur
    "psych_pairs": "using_psychometric_pairs",
}
