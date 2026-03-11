"""
Fitting and curve functions for WCTE beam analysis.

This module provides various curve functions (Gaussian, Landau-Gauss convolution)
and fitting utilities used for analyzing detector response and time-of-flight distributions.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import moyal


def gaussian(x, amp, mean, sigma):
    """
    Gaussian (normal) distribution curve.

    Parameters
    ----------
    x : float or np.ndarray
        Independent variable
    amp : float
        Amplitude (peak value)
    mean : float
        Mean of the distribution
    sigma : float
        Standard deviation (width parameter)

    Returns
    -------
    float or np.ndarray
        Gaussian value(s) at x
    """
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def three_gaussians(x, amp, mean, sigma, amp1, mean1, sigma1, amp2, mean2, sigma2):
    """
    Sum of three Gaussian distributions.

    Useful for fitting T4 TOF distributions that contain multiple particle peaks.

    Parameters
    ----------
    x : float or np.ndarray
        Independent variable
    amp, mean, sigma : float
        Amplitude, mean, and width of first Gaussian
    amp1, mean1, sigma1 : float
        Amplitude, mean, and width of second Gaussian
    amp2, mean2, sigma2 : float
        Amplitude, mean, and width of third Gaussian

    Returns
    -------
    float or np.ndarray
        Sum of the three Gaussian values at x
    """
    return (
        amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
        + amp1 * np.exp(-0.5 * ((x - mean1) / sigma1) ** 2)
        + amp2 * np.exp(-0.5 * ((x - mean2) / sigma2) ** 2)
    )


def landau_gauss_convolution(x, amp, mpv, eta, sigma):
    """
    Convolution of Landau and Gaussian distributions.

    Models the energy loss distribution (Landau) convolved with detector resolution (Gaussian).
    Used for fitting energy depositions and other physics distributions.

    Parameters
    ----------
    x : float or np.ndarray
        Independent variable (e.g., energy)
    amp : float
        Overall amplitude
    mpv : float
        Most probable value of the Landau distribution
    eta : float
        Scale parameter of the Landau distribution
    sigma : float
        Standard deviation of the Gaussian (detector resolution)

    Returns
    -------
    np.ndarray
        Convolved distribution values at x

    Notes
    -----
    Uses numerical integration to compute the convolution. The integration domain is
    automatically determined to stay within physical (positive) region and avoid
    numerical overflow in the Landau tail.
    """
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-3)
    eta = max(float(eta), 1e-3)

    # Keep the integration domain within the physical (positive) region to avoid
    # numerical overflow in the Landau tail.
    t_min = max(mpv - 5.0 * eta - 5.0 * sigma, 0.0)
    t_max = mpv + 15.0 * eta + 5.0 * sigma
    if t_max <= t_min:
        t_max = t_min + max(eta, sigma, 1.0)

    # Create integration grid
    t = np.linspace(t_min, t_max, 2000)

    # Compute Landau PDF with error handling
    with np.errstate(over="ignore", under="ignore"):
        log_pdf = moyal.logpdf(t, loc=mpv, scale=eta)

    # Clip to keep exponentiation stable
    log_pdf = np.clip(log_pdf, -700, 50)
    landau_pdf = np.exp(log_pdf)

    # Compute Gaussian for each x value
    gauss = np.exp(-0.5 * ((x[:, None] - t[None, :]) / sigma) ** 2) / (
        sigma * np.sqrt(2.0 * np.pi)
    )

    # Integrate: convolve Landau with Gaussian
    conv = np.trapz(landau_pdf * gauss, t, axis=1)

    return amp * conv


def fit_gaussian(entries, bin_centers):
    """
    Fit a Gaussian distribution to histogram data.

    Parameters
    ----------
    entries : np.ndarray
        Histogram bin values (counts)
    bin_centers : np.ndarray
        Bin center positions

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - popt : Optimized parameters [amp, mean, sigma]
        - pcov : Covariance matrix estimate

    Notes
    -----
    Initial guesses are automatically computed from the histogram:
    - amp: maximum bin value
    - mean: position of maximum bin
    - sigma: standard deviation computed from bin positions weighted by counts
    """
    amp_guess = np.max(entries)
    mean_guess = bin_centers[np.argmax(entries)]
    sigma_guess = np.std(np.repeat(bin_centers, entries.astype(int)))

    popt, pcov = curve_fit(
        gaussian,
        bin_centers,
        entries,
        p0=[amp_guess, mean_guess, sigma_guess],
    )

    return popt, pcov


def fit_three_gaussians(entries, bin_centers):
    """
    Fit three Gaussian distributions to histogram data.

    Useful for fitting multimodal distributions containing multiple particle peaks.

    Parameters
    ----------
    entries : np.ndarray
        Histogram bin values (counts)
    bin_centers : np.ndarray
        Bin center positions

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - popt : Optimized parameters [amp, mean, sigma, amp1, mean1, sigma1, amp2, mean2, sigma2]
        - pcov : Covariance matrix estimate

    Notes
    -----
    Initial guesses are automatically computed:
    - Primary peak (popt[0:3]): centered at maximum bin
    - Secondary peaks offset by ±4 bins from the primary peak
    - All amplitudes scaled relative to the primary peak
    """
    amp_guess = np.max(entries)
    mean_guess = bin_centers[np.argmax(entries)]
    sigma_guess = np.std(np.repeat(bin_centers, entries.astype(int)))

    popt, pcov = curve_fit(
        three_gaussians,
        bin_centers,
        entries,
        p0=[
            amp_guess,
            mean_guess,
            sigma_guess,
            amp_guess / 100,
            mean_guess - 4,
            sigma_guess,
            amp_guess / 100,
            mean_guess + 4,
            sigma_guess,
        ],
    )

    return popt, pcov
