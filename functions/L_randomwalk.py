import numpy as np
import xarray as xr

def random_walk(num_particles, sigma):
    rx = xr.DataArray(np.random.normal(0, sigma, num_particles), dims="particle")
    ry = xr.DataArray(np.random.normal(0, sigma, num_particles), dims="particle")
    return rx, ry

def biased_random_walk(num_particles, sigma, t, steps, bias_strength):
    # Normalised time 0 → 1
    tau = t / (steps - 1)

    # Piecewise directional bias
    if tau < 0.3:
        x_bias_direction = +1.0

    elif tau < 0.6:
        x_bias_direction = 0.0

    else:
        x_bias_direction = -1.0

    # Apply bias strength
    x_bias = bias_strength * x_bias_direction

    # Random walk with bias added to x
    rx = xr.DataArray(
        np.random.normal(loc=x_bias, scale=sigma, size=num_particles),
        dims="particle"
    )

    # y remains unbiased
    ry = xr.DataArray(
        np.random.normal(loc=0.0, scale=sigma, size=num_particles),
        dims="particle"
    )

    return rx, ry
