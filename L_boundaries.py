import numpy as np
import xarray as xr

def apply_boundaries(
        px, py, grounded,
        nx, ny, dx, dy,
        step,
        min_settle_steps,
        settlement_width=10
):
    """
    Coastline at x=0.
    Settlement zone: 0-10 km.

    Before competency:
        - particles cannot settle
        - particles that hit x<0 reflect

    After competency:
        - particles settle if they enter x<10 km
    """

    # --------------------------------
    # WESTERN COASTLINE
    # --------------------------------

    coast_hit = px < 0

    # Reflect off coastline
    px = xr.where(
        coast_hit,
        -px,
        px
    )

    # --------------------------------
    # SETTLEMENT ZONE
    # --------------------------------

    if step >= min_settle_steps:

        settle = px < settlement_width

        grounded = grounded | settle

    # --------------------------------
    # OPEN OCEAN EXITS
    # --------------------------------

    out = (
        (px > (nx - 1) * dx)
        | (py < 0)
        | (py > (ny - 1) * dy)
    )

    px = xr.where(out, np.nan, px)
    py = xr.where(out, np.nan, py)

    return px, py, grounded
