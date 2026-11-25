import numpy as np

def compute_temperature_old_old(t, a_T, b_T, Delta_cc_T, Delta_y_T, N):
    T_time = a_T * np.sin(2*np.pi/365 * t + 0.8) + b_T + Delta_cc_T/365 * t
    cell_positions = np.linspace(0, 1, N)  # 0 = south, 1 = north
    return T_time[:, None] + Delta_y_T * cell_positions[None, :]

def compute_temperature_old(t,
                        N,
                        a_T,
                        b_T,
                        t_tempmax,
                        Delta_cc_T,
                        Delta_y_T,
                        refugia_cells,
                        refugia_warming,
                        refugia_delta):

    cell_positions = np.linspace(1, 0, N)

    T_season = a_T * np.cos(2*np.pi/365 * (t - t_tempmax)) + b_T
    T_warm = (Delta_cc_T/365) * t
    T_space = Delta_y_T * cell_positions

    #no global warming refugia
    mask = np.ones(N)
    if refugia_cells is not None:
        mask[refugia_cells] = refugia_warming

    #temperature field (time, cell)
    T = T_season[:, None] + T_space[None, :] + T_warm[:, None] * mask[None, :]

    #mixing refugia
    if refugia_cells is not None and len(refugia_cells) > 0:
        T[:, refugia_cells] += refugia_delta

    return T

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