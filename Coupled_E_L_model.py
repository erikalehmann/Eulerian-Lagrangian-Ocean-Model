#%% IMPORT FUNCTIONS

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from functions.L_interpolation import interpolate_velocity_vec
from functions.L_randomwalk import biased_random_walk
from functions.L_boundaries import apply_boundaries
from functions.E_mortality import *
from functions.E_initialise import *
from functions.E_temperature import *
from parameters.basic_long import *
from parameters.basic_long import *
from functions.plot_all_plots import *

import os
import glob

# %% MODEL
t = np.arange(0, t_max + dt, dt)


def y_to_cell_index(y):
    idx = int(y // cell_width_km)
    idx = N - 1 - idx
    return max(0, min(N - 1, idx))

def run_lagrangian_and_deposit(larvae_per_cell):
    x = np.linspace(0, (nx - 1) * dx, nx)
    y = np.linspace(0, (ny - 1) * dy, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    def velocity_profile(x):
        x_km_openocean = x / 80.0
        return 0.05 + 0.95 * np.exp(-((x_km_openocean - 0.7) / 0.25) ** 2)

    u = np.zeros((nx, ny))
    v = v_mod * velocity_profile(X)

    larvae_scaled = np.where(larvae_per_cell >= larvae_threshold,
                             larvae_per_cell.astype(int),
                             0)

    num_particles = int(np.sum(larvae_scaled))
    if num_particles == 0:
        S_cells = np.zeros((steps, N))  # steps = number of Lagrangian days
        traj = np.full((1, 0, 2), np.nan)
        return S_cells, traj, X, Y, u, v

    traj = np.full((steps, num_particles, 2), np.nan)
    settle_time = np.full(num_particles, np.nan)

    # Particle arrays
    px = xr.DataArray(np.zeros(num_particles), dims="particle")
    py = xr.DataArray(np.zeros(num_particles), dims="particle")
    grounded = xr.DataArray(np.zeros(num_particles, dtype=bool), dims="particle")

    idx = 0
    for cell in range(N):
        n = larvae_scaled[cell]
        for _ in range(n):
            if idx >= num_particles:
                break
            px[idx] = 1.0 + np.random.uniform(0, 10)
            # py[idx] = (N - 0.5 - cell) * cell_width_km
            py[idx] = (N - 1 - cell) * cell_width_km + np.random.uniform(0, cell_width_km)
            idx += 1

    # Lagrangian integration
    for step in range(steps):
        up, vp = interpolate_velocity_vec(px, py, u, v, dx, dy)
        rx, ry = biased_random_walk(
            num_particles=num_particles,
            sigma=sigma,
            t=step,
            steps=steps,
            bias_strength=bias_strength
        )

        px = xr.where(~grounded, px + up * dt_L + rx, px)
        py = xr.where(~grounded, py + vp * dt_L + ry, py)
        px, py, grounded = apply_boundaries(
            px, py, grounded,
            nx, ny, dx, dy,
            step,
            min_settle_steps,
            settlement_width=10
        )

        newly_grounded = grounded.values & np.isnan(settle_time)
        settle_time[newly_grounded] = step

        traj[step, :, 0] = px.values
        traj[step, :, 1] = py.values

    # Convert grounded particles to Eulerian settlement
    S_cells = np.zeros((steps, N))  # steps = number of Lagrangian days
    for p in range(num_particles):
        if grounded.values[p] and not np.isnan(py.values[p]):
            cell = y_to_cell_index(py.values[p])
            tau = settle_time[p]
            if not np.isnan(tau) and tau >= min_settle_steps:  # tau> min_settle_steps or 0
                # if tau >= min_settle_steps:
                #    settle_day = int(tau)
                # else:
                #    settle_day = min_settle_steps
                recruits = a_set * np.exp(-b_set * (tau / steps))
                settle_day = int(tau)
                S_cells[settle_day, cell] += recruits

    # print("num_particles:", num_particles)
    # print("any grounded at end:", bool(grounded.values.any()))
    # print("S_cells after settlement:", S_cells)

    return S_cells, traj, X, Y, u, v

def run_eulerian_model(t):

    if trial_name == "spinup":
        A, R, E = initialise_populations_from_zero(t, N, input_cell)
    else:
        ds = xr.open_zarr(f"/Users/erika/PycharmProjects/Euler-Lagran/src/project/{experiment_name}/Euler_spinup.zarr")
        A, R, E = initialise_from_steady_state(ds, t, N)

    T = compute_temperature(t, N, a_T, b_T, t_tempmax,
                        Delta_cc_T, Delta_y_T,
                        refugia_cells=refugia, refugia_warming=refugia_warming)

    S = np.zeros((len(t), N))

    for k in range(len(t) - 1):

        # Once per year → spawn larvae → run Lagrangian model
        if t[k] % 365 == 0:
            larvae_per_cell = f * A[k, :]
            # print("year", k // 365, "A:", A[k, :])
            print("year", k // 365, "AN:", A[k, 0], "Amid:", A[k, 24], "AS:", A[k, -1], )
            S_cells, traj, X, Y, u, v = run_lagrangian_and_deposit(larvae_per_cell)

            # deposit recruits on correct Eulerian days
            for tau in range(steps):
                if k + tau < len(t):
                    S[k + tau, :] += S_cells[tau, :]

            year = k // 365

        # Recruits
        for i in range(N):
            R[k + 1, i] = R[k, i] + dt * (
                    S[k, i]
                    - gamma * R[k, i]
                    - mortality_R(T=T[k, i], m_min_R=m_min_R, c_R=c_R, T_opt_R=T_opt_R, T_br_R=T_br_R) * R[k, i] # #mortality_R(T[k, i])
            )

        # Adults
        for i in range(1, N - 1):
            A[k + 1, i] = A[k, i] + dt * (
                    gamma * R[k, i]
                    - mortality_A(T=T[k, i], m_min_A=m_min_A, c_A=c_A, T_opt_A=T_opt_A, T_br_A=T_br_A) * A[k, i]
                    - eta_A * A[k, i] * (A[k, i] / K_A) ** p_K
                    + D_A * (A[k, i + 1] - 2 * A[k, i] + A[k, i - 1])
            )

            # A[k+1, i] = max(0.0, A[k+1, i])
            A[k + 1, i] = np.clip(A[k + 1, i], 0, 1e6)

        A[k + 1, 0] = A[k + 1, 1]
        A[k + 1, -1] = A[k + 1, -2]

    total_steps = A.shape[0]
    if total_steps % 365 != 0:
        trim = total_steps % 365
        A = A[:-trim]
        R = R[:-trim]
        E = E[:-trim]
        T = T[:-trim]
        t = t[:-trim]

    # --- RESHAPE ---
    years = A.shape[0] // 365
    days = 365
    cells = A.shape[1]

    A_yr = A.reshape(years, days, cells)
    R_yr = R.reshape(years, days, cells)
    E_yr = E.reshape(years, days, cells)
    T_yr = T.reshape(years, days, cells)
    t_yr = t.reshape(years, days)

    # --- BUILD DATASET ---
    ds_euler = xr.Dataset(
        {
            "A": (("year", "day", "cell"), A_yr),
            "R": (("year", "day", "cell"), R_yr),
            "E": (("year", "day", "cell"), E_yr),
            "T": (("year", "day", "cell"), T_yr),
            "time_index": (("year", "day"), t_yr),
        },
        coords={
            "year": np.arange(years),
            "day": np.arange(days),
            "cell": np.arange(cells),
        }
    )

    # --- CRITICAL: CHUNK BEFORE WRITING ---
    ds_euler = ds_euler.chunk({
        "year": 10,
        "day": 365,
        "cell": 50
    })

    out_euler = f"/Users/erika/PycharmProjects/Euler-Lagran/src/project/{experiment_name}/Euler_{trial_name}.zarr"

    # --- DELETE OLD STORE ---
    import shutil, os
    if os.path.isdir(out_euler):
        shutil.rmtree(out_euler)

    # --- SAVE NEW STORE ---
    ds_euler.to_zarr(out_euler, mode="w")

    return

#%% PARAMETERS & EXPERIMENT

#sigma=0.0
v_mod=0
D_A = 0.0
#Delta_cc_T = 0.04
#refugia = [10,11,12,13, 20,21,22,23, 30,31,32,33] #None or [cells]
refugia = [40, 41, 42, 43]
#refugia_warming = 0.8 # 0=no warming, 1=same climate change warming

Refugia_warming_str = f"{refugia_warming}".replace(".", "p")


experiment_name="basic"
experiment_name=f"Refugia_CC_0p5"
experiment_name="no_v_no_diff"
#experiment_name="EAC_direction_North"
#experiment_name="Sensitivity_Test_v-E_0diff_0warm"
#experiment_name="Sensitivity_Test_D-E_no_warming"


#trial_name = "spinup"

os.makedirs(f"{experiment_name}", exist_ok=True)
#store_dir = f"{experiment_name}/Trajectories_{trial_name}_zarr"

#changing_values = [0.0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3]
#changing_values = [2,1,0,-1,-2,-3,-4]
changing_values = [0.00,0.01,0.02,0.03,0.04,0.05,0.06]
#changing_values = [0.5]

for variable_value in changing_values:
    Delta_cc_T = variable_value
    Random_Walk_str = f"{sigma:.2f}".replace(".", "p")
    Delta_str = f"{Delta_cc_T:.3f}".replace(".", "p")
    Diff_str = f"{D_A:.5f}".replace(".", "p")

    trial_name = f"spinup"
    trial_name = f"warming_{Delta_str}"
    #trial_name = f"Diff_{Diff_str}_Eddy_{Random_Walk_str}"
    #trial_name = f"v_{v_mod}_Diffusion_{Diff_str}"
    #trial_name = f"v_{v_mod}_Eddying_{Random_Walk_str}"
    #trial_name = f"40-43_warming_{Delta_str}"

    run_eulerian_model(t)

    ds_euler = xr.open_zarr(
        f"/Users/erika/PycharmProjects/Euler-Lagran/src/project/{experiment_name}/Euler_{trial_name}.zarr")
    plot_heatmap_adults(ds_euler, domain_km, trial_name)
    #plot_single_euler_cell(ds_euler, year=20, cell=8)


#%% PLOTTING
import numpy as np
import xarray as xr
import os

from functions.plot_all_plots import *
from parameters.basic_long import *

Delta_str = f"{Delta_cc_T:.3f}".replace(".", "p")
Refugia_delta_str = f"{refugia_delta}".replace(".", "p")
Refugia_warming_str = f"{refugia_warming}".replace(".", "p")

ds_euler = xr.open_zarr(f"/Users/erika/PycharmProjects/Euler-Lagran/src/project/{experiment_name}/Euler_{trial_name}.zarr")
#ds_lagran = xr.open_zarr(f"/Users/erika/PycharmProjects/Euler-Lagran/src/project/{experiment_name}/Lagran_{trial_name}.zarr")

#plot_heatmap_adults(ds_euler, domain_km, trial_name)

plot_single_euler_cell(ds_euler, year=20, cell=20)
#plot_multiple_euler_cells(ds_euler, cells=[0, 2, 10])
plot_adults_with_yearly_average(ds_euler, trial_name, cell=20)
plot_multiple_threshold_edges_annual(ds_euler, trial_name, thresholds=[1,10,100], exit_cell=48)

#plot_isotherm_speed(ds_euler, N, domain_km, trial_name)
#plot_mean_isotherm_velocity(ds_euler, domain_km, N, trial_name)
#plot_temperature_cells(ds_euler, cells=[10, 20, 30, 40])

#plot_lagrangian_from_combined(ds_lagran, N, cell_width_km, domain_km, particles_to_plot=30, plot_every_years=10)
#plot_eulerian_and_lagrangian_temperature(ds_euler, ds_lagran, cell_width_km=cell_width_km, domain_km=domain_km, plot_fraction=0.1)


