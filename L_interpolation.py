import numpy as np

def interpolate_velocity_vec(xp, yp, u, v, dx, dy, power=2):
    # mask invalid particles
    valid = ~np.isnan(xp) & ~np.isnan(yp)

    i = np.zeros_like(xp, dtype=int)
    j = np.zeros_like(yp, dtype=int)

    i[valid] = np.floor_divide(xp[valid], dx).astype(int)
    j[valid] = np.floor_divide(yp[valid], dy).astype(int)

    i = np.clip(i, 0, u.shape[0] - 2)
    j = np.clip(j, 0, u.shape[1] - 2)

    x0 = i * dx
    x1 = (i + 1) * dx
    y0 = j * dy
    y1 = (j + 1) * dy

    d00 = np.hypot(xp - x0, yp - y0)
    d10 = np.hypot(xp - x1, yp - y0)
    d01 = np.hypot(xp - x0, yp - y1)
    d11 = np.hypot(xp - x1, yp - y1)

    eps = 1e-12
    d00 = np.maximum(d00, eps)
    d10 = np.maximum(d10, eps)
    d01 = np.maximum(d01, eps)
    d11 = np.maximum(d11, eps)

    w00 = 1 / (d00 ** power)
    w10 = 1 / (d10 ** power)
    w01 = 1 / (d01 ** power)
    w11 = 1 / (d11 ** power)

    u00 = u[i, j]
    u10 = u[i+1, j]
    u01 = u[i, j+1]
    u11 = u[i+1, j+1]

    v00 = v[i, j]
    v10 = v[i+1, j]
    v01 = v[i, j+1]
    v11 = v[i+1, j+1]

    wsum = w00 + w10 + w01 + w11

    u_interp = (u00*w00 + u10*w10 + u01*w01 + u11*w11) / wsum
    v_interp = (v00*w00 + v10*w10 + v01*w01 + v11*w11) / wsum

    return u_interp, v_interp
