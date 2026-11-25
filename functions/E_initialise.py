import numpy as np

def initialise_populations_old(t, N, zero_or_steady):
    import xarray as xr

    A = np.zeros((len(t), N))
    R = np.zeros((len(t), N))
    E = np.zeros((len(t), N))

    if zero_or_steady ==0:
        A[0, :] = 0
        A[0, [2]] = 300

        R[0, :] = 300
        E[0, :] = 0

    else:
        ds = xr.open_zarr("/Users/erika/PycharmProjects/Euler-Lagran/src/project/ecosystem_combined.zarr")

        last_year = int(ds.year.values.max())

        A_final = ds["A"].sel(year=last_year).values
        R_final = ds["R"].sel(year=last_year).values
        E_final = ds["E"].sel(year=last_year).values

        A0 = A_final[-1, :].copy()
        R0 = R_final[-1, :].copy()
        E0 = E_final[-1, :].copy()

        A[0, :] = A0
        R[0, :] = R0
        E[0, :] = E0

    return A, R, E

def initialise_populations_from_zero(t,N,input_cell):
    A = np.zeros((len(t), N))
    R = np.zeros((len(t), N))
    E = np.zeros((len(t), N))

    A[0, :] = 0
    A[0, [input_cell]] = 800

    R[0, :] = 0
    E[0, :] = 0

    return A, R, E

def initialise_from_steady_state(ds,t,N):
    last_year = int(ds.year.values.max())

    A_final = ds["A"].sel(year=last_year).values
    R_final = ds["R"].sel(year=last_year).values
    E_final = ds["E"].sel(year=last_year).values

    A0 = A_final[-1, :].copy()
    R0 = R_final[-1, :].copy()
    E0 = E_final[-1, :].copy()

    A = np.zeros((len(t), N))
    R = np.zeros((len(t), N))
    E = np.zeros((len(t), N))

    A[0, :] = A0
    R[0, :] = R0
    E[0, :] = E0

    return A, R, E
