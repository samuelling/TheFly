"""Hexagonal-lattice stimuli for in-silico contrast experiments on the flyvis models."""
import numpy as np
from flyvis.utils.hex_utils import get_hex_coords, hex_to_pixel

EXTENT = 15  # 721 columns, matches the flyvis eye


def lattice(extent=EXTENT):
    u, v = get_hex_coords(extent)
    x, y = hex_to_pixel(u, v)
    # normalise so that neighbouring columns are 1 unit apart
    d = np.sqrt(x**2 + y**2)
    nn = np.min(d[d > 0])
    return u, v, x / nn, y / nn


def grating_movie(c_center, c_surround, direction_deg, *, wavelength=8.0, tf=2.0,
                  r_center=3.0, r_gap=5.0, dt=0.01, t_pre=1.0, t_stim=2.0,
                  mean=0.5, surround_phase=0.0, surround_dir_offset=0.0):
    """Drifting sinusoidal grating in a centre disk plus an annulus surround.

    Contrast is Michelson: L = mean * (1 + c * sin(...)).
    Returns array (n_frames, 1, n_columns) and the stimulus time vector.
    """
    _, _, x, y = lattice()
    r = np.sqrt(x**2 + y**2)
    center = r <= r_center
    surround = r >= r_gap
    t = np.arange(0, t_pre + t_stim, dt)
    on = (t >= t_pre).astype(float)[:, None]

    def wave(theta_deg, phase):
        th = np.deg2rad(theta_deg)
        k = 2 * np.pi / wavelength
        proj = x * np.cos(th) + y * np.sin(th)
        return np.sin(k * proj[None, :] - 2 * np.pi * tf * t[:, None] + phase)

    lum = np.full((len(t), x.size), mean)
    lum[:, center] += mean * c_center * on * wave(direction_deg, 0.0)[:, center]
    lum[:, surround] += (mean * c_surround * on *
                         wave(direction_deg + surround_dir_offset, surround_phase)[:, surround])
    return lum[:, None, :].astype(np.float32), t
