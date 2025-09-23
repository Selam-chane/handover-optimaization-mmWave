#Utility functions

import numpy as np
from sklearn.preprocessing import MinMaxScaler
_snr_scaler = MinMaxScaler(feature_range=(0, 1))


def gen_state(snr_values, fit=False, snr_min=-20, snr_max=70):
    global _snr_scaler
    snr_values = np.array(snr_values).reshape(-1, 1)
    if fit or not hasattr(_snr_scaler, "scale_"):
        ref = np.array([[snr_min], [snr_max]])
        _snr_scaler.fit(ref)
    return _snr_scaler.transform(snr_values).flatten()


def cal_reward(rates, ho_count, rate_threshold=50, max_handover_count=4):
    rates = np.array(rates, dtype=float)
    min_rate = float(np.min(rates)) if rates.size > 0 else 0.0
    half_handover_limit = max_handover_count // 2
    if min_rate > rate_threshold:
        if ho_count > half_handover_limit:
            reward = -2
        elif 0 < ho_count <= half_handover_limit:
            reward = 2 + min_rate / rate_threshold
        else:  # ho_cost == 0
            reward = -0.5
    else:
        reward = -2 + min_rate / rate_threshold
    return reward


def action_space(num_cells=25):
    return np.arange(num_cells)


