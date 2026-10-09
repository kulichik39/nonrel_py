import numpy as np
import math
from scipy.special import legendre
from scipy.optimize import curve_fit
from fortran_output_analysis.common_utility import assert_abs_or_emi

"""
This namespace contains functions for analyzing complex q parameter in the one and 
two photon cases.
"""


def get_epsilon(en, abs_or_emi, g_omega_IR, E_res, width_res):
    """
    Calculates an array of resclased energies eps.

    Args:
        en - array of energies.
        abs_or_emi - tells if the resonance was in the absorption or emission path, must be
                     the "abs" or "emi" string.
        g_omega_IR - energy of the IR photon.
        E_res - energy of the resonance.
        width_res - width of the resonance.
        NOTE: en, g_omega_IR, E_res, width_res must be in the same units.

    Returns:
        eps - an array of rescaled energies.
    """

    assert_abs_or_emi(abs_or_emi)

    if abs_or_emi == "abs":
        eps = (en - (E_res + g_omega_IR)) / (width_res / 2)
    else:
        eps = (en - (E_res - g_omega_IR)) / (width_res / 2)

    return eps


def get_coefficients_for_integrated(
    abs_or_emi,
    eps,
    integrated_intensity,
    q,
    eps_1=1.0,
    eps_2=0.0,
):
    """
    Calculates the A_bg and A_res coefficients in the angularly integrated case.

    Args:
        abs_or_emi - tells if the resonance was in the absorption or emission path, must be
                     the "abs" or "emi" string.
        eps - array of rescaled energies.
        integrated_intensity - array of integrated intensities.
        q - Fano shape parameter.
        eps_1 - the first rescaled energy point to estimate the coefficients.
        eps_2 - the second rescaled energy point to estimate the coefficients.
        NOTE: the output is highly sensitive to the choice of eps_1 and eps_2!

    Returns:
        A_bg, A_res - model coefficients.
    """

    assert_abs_or_emi(abs_or_emi)

    # find intenisties at the sepcified points
    intens_1 = np.interp(eps_1, eps, integrated_intensity)
    intens_2 = np.interp(eps_2, eps, integrated_intensity)
    # If the resonance was in the absorption path then the model describes conjugated intensity
    if abs_or_emi == "abs":
        intens_1 = np.conjugate(intens_1)
        intens_2 = np.conjugate(intens_2)

    # compute the coefficients
    A_res = (intens_2 - intens_1) / (
        (eps_2 + q) / (eps_2 + 1 * 1j) - (eps_1 + q) / (eps_1 + 1 * 1j)
    )
    A_bg = intens_1 - A_res * (eps_1 + q) / (eps_1 + 1 * 1j)

    return A_bg, A_res


def get_complex_q_integrated(A_bg, A_res, q):
    """
    Computes complex q parameter in the angularly integrated case.

    Args:
        A_bg, A_res - model coefficients.
        q - Fano shape parameter.

    Returns:
        the complex q parameter.
    """

    if A_res == 0 + 0 * 1j and A_bg == 0 + 0 * 1j:
        return 0

    return (A_res * q + 1j * A_bg) / (A_res + A_bg)


def get_background_contribution_integrated(abs_or_emi, A_bg, A_res):
    """
    Computes the background contribution (the energy independent term) in the angularly
    integrated case.

    Args:
        abs_or_emi - tells if the resonance was in the absorption or emission path, must be
                     the "abs" or "emi" string.
        A_bg, A_res - model coefficients.

    Returns:
        background_cont - the background contribution.
    """

    assert_abs_or_emi(abs_or_emi)

    background_cont = A_res + A_bg
    if abs_or_emi == "abs":
        background_cont = np.conjugate(background_cont)

    return background_cont


def get_coefficients_for_beta_param(
    abs_or_emi,
    eps,
    integrated_intensity,
    beta_param,
    q,
    eps_1=1.0,
    eps_2=0.0,
):
    """
    Finds the A_bg and A_res coefficients for the given beta parameter.

    Args:
        abs_or_emi - tells if the resonance was in the absorption or emission path, must be
                     the "abs" or "emi" string.
        eps - array of rescaled energies.
        integrated_intensity - array of integrated intensities.
        beta_param - array of beta parameter values.
        q - Fano shape parameter.
        eps_1 - the first rescaled energy point to estimate the coefficients.
        eps_2 - the second rescaled energy point to estimate coefficients.
        NOTE: the output is highly sensitive to the choice of eps_1 and eps_2!

    Returns:
        A_bg, A_res - coefficients for the beta parameter.
    """

    assert_abs_or_emi(abs_or_emi)

    # extract the numertor in the formula for the beta parameter
    beta_numerator = beta_param * integrated_intensity

    # find the beta numerator at the specified points
    beta_numerator_1 = np.interp(eps_1, eps, beta_numerator)
    beta_numerator_2 = np.interp(eps_2, eps, beta_numerator)

    # If the resonance was in the absorption path then the model describes conjugated parameter
    if abs_or_emi == "abs":
        beta_numerator_1 = np.conjugate(beta_numerator_1)
        beta_numerator_2 = np.conjugate(beta_numerator_2)

    # compute the coefficients
    A_res = (beta_numerator_2 - beta_numerator_1) / (
        (eps_2 + q) / (eps_2 + 1 * 1j) - (eps_1 + q) / (eps_1 + 1 * 1j)
    )
    A_bg = beta_numerator_1 - A_res * (eps_1 + q) / (eps_1 + 1 * 1j)

    return A_bg, A_res


def get_complex_q_angular(
    angle, A_bg_int, A_res_int, A_bg_b2, A_res_b2, A_bg_b4, A_res_b4, q
):
    """
    Computes the complex q parameter in the angularly resolved case.

    Args:
        angle - emission angle.
        A_bg_int, A_res_int - coefficients for the integrated case.
        A_bg_b2, A_res_b2 - coefficients for the beta parameter of order 2.
        A_bg_b4, A_res_b4 - coefficients for the beta parameter of order 4 (NOTE: set them to zero if
                            you work with the one photon case).
        q - Fano shape parameter.

    Returns:
        value of the complex q parameter for the given emission angle.
    """

    # get the corresponding parameters
    q_complex_int = get_complex_q_integrated(A_bg_int, A_res_int, q)
    q_complex_b2 = get_complex_q_integrated(A_bg_b2, A_res_b2, q)
    q_complex_b4 = get_complex_q_integrated(A_bg_b4, A_res_b4, q)

    # Introduce some auxiliary values for calculations
    A_int = A_res_int + A_bg_int
    A_b2 = (A_res_b2 + A_bg_b2) * legendre(2)(np.array(np.cos(math.radians(angle))))
    A_b4 = (A_res_b4 + A_bg_b4) * legendre(4)(np.array(np.cos(math.radians(angle))))

    return (A_int * q_complex_int + A_b2 * q_complex_b2 + A_b4 * q_complex_b4) / (
        A_int + A_b2 + A_b4
    )


def get_background_contribution_angular(
    angle, abs_or_emi, A_bg_int, A_res_int, A_bg_b2, A_res_b2, A_bg_b4, A_res_b4
):
    """
    Computes the background contribution (the energy independent term) in the angularly
    resolved case.

    Args:
        angle - emission angle.
        abs_or_emi - tells if the resonance was in the absorption or emission path, must be
                     the "abs" or "emi" string.
        A_bg_int, A_res_int - coefficients for the integrated case.
        A_bg_b2, A_res_b2 - coefficients for the beta parameter of order 2.
        A_bg_b4, A_res_b4 - coefficients for the beta parameter of order 4 (NOTE: set them to zero if
                            you work with the one photon case).

    Returns:
        background_cont - value of the background contribution for the give angle.
    """

    assert_abs_or_emi(abs_or_emi)

    # Introduce some auxiliary values for calculations
    A_int = A_res_int + A_bg_int
    A_b2 = (A_res_b2 + A_bg_b2) * legendre(2)(np.array(np.cos(math.radians(angle))))
    A_b4 = (A_res_b4 + A_bg_b4) * legendre(4)(np.array(np.cos(math.radians(angle))))

    background_cont = A_int + A_b2 + A_b4
    if abs_or_emi == "abs":
        background_cont = np.conjugate(background_cont)

    return background_cont


def get_zeros(q_complex):
    """
    Computes zeros of the real and imaginary parts of the energy dependent term.
    Works for both cases: angularly integrated and angularly resolved.

    Args:
        q_complex - complex q parameter (either from the integrated or resolved data).

    Returns:
        im_zero - zero of the imaginary part.
        re_zero_1 - the first zero of the real part.
        re_zero_2 - the second zero of the real part.
    """

    im_zero = np.real(q_complex) / (np.imag(q_complex) - 1)
    re_zero_1 = -np.real(q_complex) / 2 + np.sqrt(
        (np.real(q_complex) / 2) ** 2 - np.imag(q_complex)
    )
    re_zero_2 = -np.real(q_complex) / 2 - np.sqrt(
        (np.real(q_complex) / 2) ** 2 - np.imag(q_complex)
    )

    return im_zero, re_zero_1, re_zero_2


def get_coefficients_for_integrated_by_fit(
    abs_or_emi, q, int_intens, eps, eps_min, eps_max
):
    """
    Fits the model to the provided range of an integrated intensity and retrieves the coefficients.

    Args:
        abs_or_emi - tells if the resonance was in the absorption or emission path, must be
                     the "abs" or "emi" string.
        q - Fano shape parameter.
        int_intens - array of the integrated intensities.
        eps - array of reduced energies.
        eps_min, eps_max - the lower and upper boundaries of the range to fit.

    Returns:
        A_bg, A_res - model coefficients.
        pcov - covariance matrix containing the errors of the fitting.
    """

    assert_abs_or_emi(abs_or_emi)

    # extcact the fitting range
    mask_fit = np.logical_and(eps >= eps_min, eps <= eps_max)
    eps_fit = eps[mask_fit]
    int_intens_fit = int_intens[mask_fit]

    def func_fit(x, A_res_Re, A_res_Im, A_bg_Re, A_bg_Im):
        """The fitting function."""

        if abs_or_emi == "abs":
            return np.conjugate(
                (A_res_Re + 1j * A_res_Im) * (x + q) / (x + 1j * 1)
                + A_bg_Re
                + 1j * A_bg_Im
            )

        else:
            return (
                (A_res_Re + 1j * A_res_Im) * (x + q) / (x + 1j * 1)
                + A_bg_Re
                + 1j * A_bg_Im
            )

    def func_fit_flatten(x, A_res_Re, A_res_Im, A_bg_Re, A_bg_Im):
        """
        The helper that flattens the complex fitting function to an array of a doubled size,
        containing the real and imaginary parts separately.
        Needed since the scipy's curve fit doesn't work with complex values.
        """
        # x is the double size vector composed of two identical arrays with the original points,
        # i.e x = [x, x]
        N = len(x)
        x_real = x[: N // 2]
        x_imag = x[N // 2 :]
        y_real = np.real(func_fit(x_real, A_res_Re, A_res_Im, A_bg_Re, A_bg_Im))
        y_imag = np.imag(func_fit(x_imag, A_res_Re, A_res_Im, A_bg_Re, A_bg_Im))

        return np.hstack([y_real, y_imag])

    y_true_real = np.real(int_intens_fit)
    y_true_imag = np.imag(int_intens_fit)

    y_true_flatten = np.hstack([y_true_real, y_true_imag])

    popt, pcov = curve_fit(
        func_fit_flatten, np.hstack([eps_fit, eps_fit]), y_true_flatten
    )

    A_res_Re, A_res_Im, A_bg_Re, A_bg_Im = popt

    A_bg = A_bg_Re + 1j * A_bg_Im
    A_res = A_res_Re + 1j * A_res_Im

    return A_bg, A_res, pcov


def get_coefficients_for_beta_param_by_fit(
    abs_or_emi, q, beta_param, eps, eps_min, eps_max, A_bg_int, A_res_int
):
    """
    Fits the model to the provided range of a beta parameter and retrieves the coefficients.

    Args:
        abs_or_emi - tells if the resonance was in the absorption or emission path, must be
                     the "abs" or "emi" string.
        q - Fano shape parameter.
        beta_param - array of the beta parameter values.
        eps - array of reduced energies.
        eps_min, eps_max - the lower and upper boundaries of the range to fit.
        A_bg_int, A_res_int - coefficients for the integrated case.

    Returns:
        A_bg, A_res - model coefficients.
        pcov - covariance matrix containing the errors of the fitting.
    """

    assert_abs_or_emi(abs_or_emi)

    # extcact the fitting range
    mask_fit = np.logical_and(eps >= eps_min, eps <= eps_max)
    eps_fit = eps[mask_fit]
    beta_param_fit = beta_param[mask_fit]

    def func_fit(x, A_res_Re, A_res_Im, A_bg_Re, A_bg_Im):
        """The fitting function."""

        if abs_or_emi == "abs":
            return np.conjugate(
                (
                    (A_res_Re + 1j * A_res_Im) * (x + q) / (x + 1j * 1)
                    + A_bg_Re
                    + 1j * A_bg_Im
                )
                / (A_res_int * (x + q) / (x + 1j * 1) + A_bg_int)
            )

        else:
            return (
                (A_res_Re + 1j * A_res_Im) * (x + q) / (x + 1j * 1)
                + A_bg_Re
                + 1j * A_bg_Im
            ) / (A_res_int * (x + q) / (x + 1j * 1) + A_bg_int)

    def func_fit_flatten(x, A_res_Re, A_res_Im, A_bg_Re, A_bg_Im):
        """
        The helper that flattens the complex fitting function to an array of a doubled size,
        containing the real and imaginary parts separately.
        Needed since the scipy's curve fit doesn't work with complex values.
        """
        # x is the double size vector composed of two identical arrays with the original points,
        # i.e x = [x, x]
        N = len(x)
        x_real = x[: N // 2]
        x_imag = x[N // 2 :]
        y_real = np.real(func_fit(x_real, A_res_Re, A_res_Im, A_bg_Re, A_bg_Im))
        y_imag = np.imag(func_fit(x_imag, A_res_Re, A_res_Im, A_bg_Re, A_bg_Im))

        return np.hstack([y_real, y_imag])

    y_true_real = np.real(beta_param_fit)
    y_true_imag = np.imag(beta_param_fit)

    y_true_flatten = np.hstack([y_true_real, y_true_imag])

    popt, pcov = curve_fit(
        func_fit_flatten, np.hstack([eps_fit, eps_fit]), y_true_flatten
    )

    A_res_Re, A_res_Im, A_bg_Re, A_bg_Im = popt

    A_bg = A_bg_Re + 1j * A_bg_Im
    A_res = A_res_Re + 1j * A_res_Im

    return A_bg, A_res, pcov


def get_model_pred_for_integrated(abs_or_emi, eps, A_bg, A_res, q_complex):
    """
    Calculates model prediction for an angularly integrated intensity.

    Args:
        abs_or_emi - tells if the resonance was in the absorption or emission path, must be
                     the "abs" or "emi" string.
        eps - array of reduced energies.
        A_bg, A_res - model coefficients.
        q_complex - complex q parameter.

    Returns:
        model prediction for the angularly integrated intensity.
    """

    assert_abs_or_emi(abs_or_emi)

    # background contribution
    background_contr = get_background_contribution_integrated(abs_or_emi, A_bg, A_res)

    # energy-dependent term
    energy_contr = (eps + q_complex) / (eps + 1 * 1j)
    if abs_or_emi == "abs":  # conjugate for the absroption path
        energy_contr = np.conjugate(energy_contr)

    return energy_contr * background_contr


def get_model_pred_for_beta_param(
    abs_or_emi,
    eps,
    A_bg_beta,
    A_res_beta,
    q_complex_beta,
    A_bg_int,
    A_res_int,
    q_complex_int,
):
    """
    Calculates model prediction for a beta parameter.

    Args:
        abs_or_emi - tells if the resonance was in the absorption or emission path, must be
                     the "abs" or "emi" string.
        eps - array of reduced energies.
        A_bg_beta, A_res_beta - coefficients for the beta parameter.
        q_complex_beta - complex q for the beta parameter.
        A_bg_int, A_res_int - coefficients for the integrated intensity.
        q_complex_int - complex q for the integrated intensity.

    Returns:
        model prediction for the beta parameter.
    """

    assert_abs_or_emi(abs_or_emi)

    model_pred = (
        (A_res_beta + A_bg_beta)
        * (eps + q_complex_beta)
        / ((A_res_int + A_bg_int) * (eps + q_complex_int))
    )

    if abs_or_emi == "abs":
        model_pred = np.conjugate(model_pred)

    return model_pred


def get_model_pred_for_angular(
    abs_or_emi,
    eps,
    angle,
    A_bg_int,
    A_res_int,
    A_bg_b2,
    A_res_b2,
    A_bg_b4,
    A_res_b4,
    q_complex_ang,
):
    """
    Calculates model prediction for an angularly resolved intensity.

    Args:
        abs_or_emi - tells if the resonance was in the absorption or emission path, must be
                     the "abs" or "emi" string.
        eps - array of reduced energies.
        angle - emission angle.
        A_bg_int, A_res_int - coefficients for the integrated case.
        A_bg_b2, A_res_b2 - coefficients for the beta parameter of order 2.
        A_bg_b4, A_res_b4 - coefficients for the beta parameter of order 4 (NOTE: set them to zero if
                            you work with the one photon case).
        q_complex_ang - complex q parameter for the given emission angle.

    Returns:
        model prediction for the angularly resolved intensity.
    """

    assert_abs_or_emi(abs_or_emi)

    background_contr = get_background_contribution_angular(
        angle,
        abs_or_emi,
        A_bg_int,
        A_res_int,
        A_bg_b2,
        A_res_b2,
        A_bg_b4,
        A_res_b4,
    )

    energy_contr = (eps + q_complex_ang) / (eps + 1j)

    if abs_or_emi == "abs":
        energy_contr = np.conjugate(energy_contr)

    return energy_contr * background_contr


def get_crit_angles(
    angles,
    q,
    A_bg_int,
    A_res_int,
    A_bg_b2,
    A_res_b2,
    A_bg_b4,
    A_res_b4,
):
    """
    Retrieves the critical angles by looking where the imaginary part of the complex q parameter
    crosses zero.

    Args:
        angles - array of angles to search in.
        q - Fano shape parameter.
        A_bg_int, A_res_int - coefficients for the integrated case.
        A_bg_b2, A_res_b2 - coefficients for the beta parameter of order 2.
        A_bg_b4, A_res_b4 - coefficients for the beta parameter of order 4 (NOTE: set them to zero if
                            you work with the one photon case).

    Returns:
        crit_angles - array of critical angles if any.
        complex_q_arr - array of the complex q values.
    """

    complex_q_arr = []  # array to store the values of the complex q parameter

    # fill the array with complex q
    for angle in angles:
        complex_q = get_complex_q_angular(
            angle, A_bg_int, A_res_int, A_bg_b2, A_res_b2, A_bg_b4, A_res_b4, q
        )
        complex_q_arr.append(complex_q)

    complex_q_arr = np.array(complex_q_arr)

    # search for the critical angles
    complex_q_imag_arr = np.imag(complex_q_arr)  # extract the imaginary part
    crit_angles = []
    for i in range(1, len(complex_q_imag_arr)):
        if np.sign(complex_q_imag_arr[i]) != np.sign(complex_q_imag_arr[i - 1]):
            crit_angle = (angles[i] + angles[i - 1]) / 2
            crit_angles.append(crit_angle)

    crit_angles = np.array(crit_angles)

    return crit_angles, complex_q_arr


def get_crit_angles_analyt(q, resArr, bgArr, wigner=False):
    """
    Leon's implementation of the analytical way to find critical angles.

    Args:
        q - Fano shape parameter.
        resArr - array of the resonant coefficients A_res for the integerated case and for the beta
                 parameters. The order is [A_res_int, A_res_b2, A_res_b4].
                 NOTE: in the one photon (Wigner) case, the last element should be zero.
        bgArr - array of the background coefficients A_bg for the integerated case and for the beta
                parameters. The order is [A_bg_int, A_bg_b2, A_bg_b4].
                NOTE: in the one photon (Wigner) case, the last element should be zero.
        wigner - specifies either the one photon case (wigner=True) or the two photon case (wigner=False).

    Returns:
        array of critical angles if any.

    """
    atomic = not wigner
    # Please make the arrays resArr and bgArr length 3.
    # In case of Wigner, make the last value of each be zero.
    ABis = [q * res + 1j * bg for res, bg in zip(resArr, bgArr)]
    As = np.real(ABis)
    Bs = np.imag(ABis)
    CDis = [res + bg for res, bg in zip(resArr, bgArr)]
    Cs = np.real(CDis)
    Ds = np.imag(CDis)

    Ess = np.outer(As, -Ds) + np.outer(Bs, Cs)
    Fss = Ess + Ess.T

    G = 64 * Ess[0][0] + 16 * Ess[1][1] - 32 * Fss[0][1]
    H = -96 * Ess[1][1] + 96 * Fss[0][1]
    I = 144 * Ess[1][1]
    J = 0
    K = 0

    if atomic:
        G += 9 * Ess[2][2] + 24 * Fss[0][2] - 12 * Fss[1][2]
        H += -180 * Ess[2][2] - 240 * Fss[0][2] + 156 * Fss[1][2]
        I += 1110 * Ess[2][2] + 280 * Fss[0][2] - 500 * Fss[1][2]
        J += -2100 * Ess[2][2] + 420 * Fss[1][2]
        K += 1225 * Ess[2][2]

    solutions = np.roots([K, J, I, H, G])

    return [
        180 / np.pi * math.acos(np.sqrt(solution))
        for solution in solutions
        if abs(solution.imag) < 1e-12 and solution.real < 1 and solution.real > 0
    ]
