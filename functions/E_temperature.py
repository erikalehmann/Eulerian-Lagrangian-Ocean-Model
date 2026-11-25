import numpy as np

def compute_temperature(t, N, a_T, b_T, t_tempmax,
                        Delta_cc_T, Delta_y_T,
                        refugia_cells, refugia_warming):

    cell_positions = np.linspace(1, 0, N)
    T_season = a_T * np.cos(2*np.pi/365 * (t - t_tempmax)) + b_T
    T_space = Delta_y_T * cell_positions
    T_warm = (Delta_cc_T / 365) * t

    if refugia_cells is not None:
        mask = np.ones(N)
        mask[refugia_cells] = refugia_warming
    else:
        mask = np.ones(N)

    T = T_season[:, None] + T_space[None, :] + T_warm[:, None] * mask[None, :]

    return T
