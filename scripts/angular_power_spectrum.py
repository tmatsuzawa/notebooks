"""
Description: Computes an angular power spectrum from a given 3D scalar field
Author: Takumi Matsuzawa
"""

import numpy as np
import numpy.ma as ma
from scipy.stats import binned_statistic, binned_statistic_2d
import scipy.special as sp
import matplotlib.pyplot as plt

Ylm = sp.sph_harm  # Ylm(m, l, phi, theta)

def cart2sph(x, y, z, periodic=False):
    """
    Transformation: cartesian to spherical
    z = r cos theta
    y = r sin theta sin phi
    x = r sin theta cos phi

    Parameters
    ----------
    x: numpy array, x-coord (Cartesian)
    y: numpy array, y-coord (Cartesian)
    z: numpy array, z-coord (Cartesian)
    periodic: bool, If True, it respects the periodic boundary condition.

    Returns
    -------
    r: radial distance
    theta: polar angle [0, pi] (angle from the z-axis)
    phi: azimuthal angle [-pi, pi] (angle on the x-y plane)
    """
    if periodic:
        lx, ly, lz = np.nanmax(x) - np.nanmin(x), np.nanmax(y) - np.nanmin(y), np.nanmax(z) - np.nanmin(z)
        x = x - lx * np.round(x / lx)
        y = y - ly * np.round(y / ly)
        z = z - lz * np.round(z / lz)

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def get_binned_stats2d(x, y, var, n_bins=100, nx_bins=None, ny_bins=None, bin_center=True,
                       xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Make a histogram out of a pair of 1d arrays.
    ... Returns arg_bins, var_mean, var_err
    ... The given arrays could contain nans and infs. They will be ignored.

    Parameters
    ----------
    x: 2d array, control variable
    y: 2d array, control variable
    var: 2d array, data array to be binned
    n_bins: int, default: 100
    mode: str, deafult: 'linear'
        If 'linear', var will be sorted to equally spaced bins. i.e. bin centers increase linearly.
        If 'log', the bins will be not equally spaced. Instead, they will be equally spaced in log.
        ... bin centers will be like... 10**0, 10**0.5, 10**1.0, 10**1.5, ..., 10**9

    Returns
    -------
    xx_binned: 2d array, bin centers about x
    yy_binned: 2d array, bin centers about y
    var_mean: 2d array,  mean values of data in each bin
    var_err: 2d array, standard error of data in each bin

    """

    def get_mask_for_nan_and_inf(U):
        """
        Returns a mask for nan and inf values in a multidimensional array U
        Parameters
        ----------
        U: N-d array

        Returns
        -------

        """
        U = np.array(U)
        U_masked_invalid = ma.masked_invalid(U)
        return U_masked_invalid.mask

    x, y, var = np.asarray(x), np.asarray(y), np.asarray(var)

    # make sure rr and corr do not contain nans
    mask_x = get_mask_for_nan_and_inf(x)
    mask_x = ~mask_x
    mask_y = get_mask_for_nan_and_inf(y)
    mask_y = ~mask_y
    mask_var = get_mask_for_nan_and_inf(var)
    mask_var = ~mask_var
    mask = mask_x * mask_y * mask_var

    if xmin is not None:
        mask_x_less = x > xmin
        mask *= mask_x_less
    if xmax is not None:
        mask_x_greater = x < xmax
        mask *= mask_x_greater
    if ymin is not None:
        mask_y_less = y > ymin
        mask *= mask_y_less
    if ymax is not None:
        mask_y_greater = y < ymax
        mask *= mask_y_greater
    if nx_bins is None and ny_bins is None:
        bins = [n_bins, n_bins]
    else:
        bins = [ny_bins, nx_bins]

    # get a histogram
    var_mean, y_edge, x_edge, binnumber = binned_statistic_2d(y[mask], x[mask], var[mask], statistic='mean', bins=bins)
    var_std, _, _, _ = binned_statistic_2d(y[mask], x[mask], var[mask], statistic='std', bins=bins)
    counts, _, _, _ = binned_statistic_2d(y[mask], x[mask], var[mask], statistic='count', bins=bins)
    var_err = var_std / np.sqrt(counts)

    if bin_center:
        binwidth_x = (x_edge[1] - x_edge[0])
        binwidth_y = (y_edge[1] - y_edge[0])
        bin_centers_x = x_edge[1:] - binwidth_x / 2
        bin_centers_y = y_edge[1:] - binwidth_y / 2
        arg_bins_x = bin_centers_x
        arg_bins_y = bin_centers_y
    else:
        arg_bins_x = x_edge[:-1]
        arg_bins_y = y_edge[:-1]

    xx_binned, yy_binned = np.meshgrid(arg_bins_x, arg_bins_y)

    return xx_binned, yy_binned, var_mean, var_err


def get_dist_on_spherical_surface(qty, xxx, yyy, zzz, R, dR=5, n_bins=100):
    """
    Returns a field on the spherical surface by averaging a given quantity within the shell

    qty: 3d array, a scalar field
    xxx: 3d array, x-coordinate
    yyy: 3d array, y-coordinate
    zzz: 3d array, z-coordinate
    R: float, radius that defines a spherical 'shell' to probe the surface distribution.
    ... A shell is defined by [R, R+dR]
    dR: float, radius that defines a spherical 'shell' to probe the surface distribution
    n_bins: int, number of bins, default: 100

    """
    rrr, tttheta, ppphi = cart2sph(xxx, yyy, zzz)

    keep = (rrr >= R) & (rrr < (R + dR))

    qty_keep, theta_keep, phi_keep = qty[keep], tttheta[keep], ppphi[keep]
    qty_avg = np.nanmean(qty_keep)

    th_bin, ph_bin, qty_bin, qty_bin_err = get_binned_stats2d(theta_keep, phi_keep, qty_keep, n_bins=n_bins)
    return th_bin, ph_bin, qty_bin, qty_bin_err


def compute_alm(qty, theta, phi, lmax=10):
    """
    Returns a coefficient of spherical harmonic expansion up to lmax
    Parameters:
    qty: 2d array, a scalar field on a spherical surface defined by (theta: polar angle, phi: azimuthal angle)
    theta: 2d array, azimuthal angle
    phi: 2d array, azimuthal angle

    Returns:
    alm: dict, a nested dictionary that stores coefficients of sphercal harmonic expansion of a given field
    ... A(theta, phi) = \int alm Ylm(theta, phi) sin(theta) dtheta dphi   where Ylm is spherical hamornics.
    ... alm = \int A(theta, phi) Ylm* (theta, phi) sin(theta) dtheta dphi
    ... alm can be retrieved by 'alm[l][m]' where 'l' and 'm' are integers. m \in [-l, l]
    lmax: int, maximum ell to compute alm. A higher l means a smaller angular resolution.
    """
    alm = {}
    dtheta, dphi = abs(theta[0, 1] - theta[0, 0]), abs(phi[1, 0] - phi[0, 0])
    for l in tqdm(range(0, lmax)):
        alm[l] = {}
        for m in range(-l, l + 1):
            integrand = qty * np.conj(Ylm(m, l, phi, theta)) * np.sin(
                theta) * dtheta * dphi  # m, l, phi (azimuthal), theta (polar)
            integral = np.nansum(integrand)
            alm[l][m] = integral
    return alm


def compute_cl(qty, theta, phi, lmax=10):  # cl is just an average of |alm|^2 over l
    """
    Returns cl:= 1/(2l+1) * sum{alm^2}

    Parameters:
    qty: 2d array, a scalar field on a spherical surface defined by (theta: polar angle, phi: azimuthal angle)
    theta: 2d array, azimuthal angle
    phi: 2d array, azimuthal angle
    lmax: int, maximum ell to compute alm. A higher l means a smaller angular resolution.

    Returns:
    cl: list, cl := 1/(2l+1) * sum{ |alm|^2 }
    ... A(theta, phi) = \int alm Ylm(theta, phi) sin(theta) dtheta dphi   where Ylm is spherical hamornics.
    ... alm = \int A(theta, phi) Ylm* (theta, phi) sin(theta) dtheta dphi
    ... alm can be retrieved by 'alm[l][m]' where 'l' and 'm' are integers. m \in [-l, l]
    """
    alm = compute_alm(qty, theta, phi, lmax=lmax)
    cl = []
    for l in range(0, lmax):
        partial_sum = 0
        for m in range(-l, l + 1):
            partial_sum += np.conjugate(alm[l][m]) * alm[l][m]
        partial_sum = float(partial_sum)
        val = partial_sum / (2 * l + 1)
        cl.append(val)
    return cl


def compute_dl(qty, theta, phi, lmax=10):  # l(2l+1)/(4pi) cl
    """
    Returns dl:= l(2l+1)/(4pi) * cl
    ... dl represents the angular power (energy per unit angle) of a quantity.
    ... Energy = \int dl(theta, phi) sin(theta) dtheta dphi

    Parameters:
    qty: 2d array, a scalar field on a spherical surface defined by (theta: polar angle, phi: azimuthal angle)
    theta: 2d array, azimuthal angle
    phi: 2d array, azimuthal angle
    lmax: int, maximum ell to compute alm. A higher l means a smaller angular resolution.

    Returns:
    dl: list, dl := l(2l+1)/(4pi) * cl
    ... cl := 1/(2l+1) * sum{ |alm|^2 }
    ... A(theta, phi) = \int alm Ylm(theta, phi) sin(theta) dtheta dphi   where Ylm is spherical hamornics.
    ... alm = \int A(theta, phi) Ylm* (theta, phi) sin(theta) dtheta dphi
    ... alm can be retrieved by 'alm[l][m]' where 'l' and 'm' are integers. m \in [-l, l]
    """
    cl = compute_cl(qty, theta, phi, lmax=lmax)
    dl = [val * l * (2 * l + 1) / 4 / np.pi for l, val in enumerate(cl, start=1)]
    return dl


def plot_mollweide(theta, phi, data, fignum=1, subplot=111, figsize=None, cmap='RdBu_r', **kwargs):
    """
    Plots a heatmap with a mollweide projection
    Parameters
    ----------
    theta: 2d array, range:0-pi, polar angle
    phi: 2d array, range: 0-2pi, azimuthal angle
    data: 2d array
    fignum: int, figure number
    subplot: int, 3-digit code for matplotlib
    figsize: tuple, figure size in inches
    cmap: str or colormap object
    kwargs: passed to color_plot()

    Returns
    -------
    fig, ax, mappable
    """
    if np.nanmax(phi) > np.pi:
        # Convert phi (0-2pi) to longitude (-pi, pi)
        longitude = phi - np.pi
    else:
        longitude = phi  # longitude must be between [-pi, pi]

    # Convert theta (0-pi) to the range [π/2, -π/2]
    latitude = -theta + np.pi / 2

    fig = plt.figure(num=fignum, figsize=figsize)
    ax = fig.add_subplot(subplot, projection='mollweide')

    mappable = ax.pcolormesh(longitude, latitude, data, cmap=cmap, **kwargs)
    ax.xaxis.set_visible(False)
    ax.set_yticks(np.linspace(-np.pi / 2, np.pi / 2, 7))
    return fig, ax, mappable

if __name__ == "__main__":
    # Create a grid
    n = 100
    x, y, z = np.linspace(-np.pi, np.pi, n), np.linspace(-np.pi, np.pi, n), np.linspace(-np.pi, np.pi, n)
    xx, yy, zz = np.meshgrid(x, y, z)
    # Cartersian to Spherical coordinates
    r, theta, phi = cart2sph(xx, yy, zz)
    # Create data
    qty3d = Ylm(-2, 2, phi, theta)  # Ylm(m, l, theta, phi)

    # Grab data within a shell defined by [R, R+dR]
    theta, phi, qty, qty_err = get_dist_on_spherical_surface(qty3d, xx, yy, zz, np.pi * 0.5, dR=np.pi * 0.5)

    # Compute angular spectrum
    dl = compute_dl(qty, theta, phi, lmax=10)
    l = list(range(10))

    # PLOT
    # ax1: Plot data with Mollweide projection
    fig, ax1, mappable = plot_mollweide(theta, phi, qty.real, subplot=211)
    ax2 = fig.add_subplot(212)

    # ax2: angular spectrum
    ax2.scatter(l, dl)
    ax2.set_xlabel('Multipole, $\ell$', fontsize=8)
    ax2.set_ylabel('Angular power, $\\frac{\ell}{4\pi}\sum_{m=-\ell}^{\ell}|a_{\ell, m}|^2$', fontsize=8)
    plt.show()