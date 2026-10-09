import numpy as np
from math import radians
from scipy.special import sph_harm_y
from sympy.physics.wigner import wigner_3j

from fortran_output_analysis.constants_and_parameters import fine_structure, au_to_Mbarn
from fortran_output_analysis.types import ArrFloat64, ArrFloat64_2D, ArrComplex128_2D
from fortran_output_analysis.common_utility import l_to_int
from fortran_output_analysis.common_utility import wavenumber
from fortran_output_analysis.onephoton.onephoton import OnePhoton, final_ls
from fortran_output_analysis.onephoton.utilities import (
    get_ekin_eV,
    get_ekin_Hartree,
    get_omega_Hartree,
    get_matrix_elements_with_coulomb_phase,
)


def get_integrated_photoion_cs_for_channel(
    one_photon: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    final_l: int | str,
) -> tuple[ArrFloat64, ArrFloat64]:
    """
    Extracts the integrated photoionization cross section for the specified ionization channel of
    the hole.

    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        final_l - orbital angular momentum of the final state, in the int or str format.

    Returns:
        ekin_eV - an array of electron kinetic energies in eV.
        cs - integrated photoionization cross section for the specified ionization channel of
             the hole.
    """

    ekin_eV = get_ekin_eV(one_photon, n_qn, hole_l)

    channels = one_photon.get_channels_for_hole(n_qn, hole_l)
    cs = channels.get_cs_one_channel(final_l)

    return ekin_eV, cs


def get_integrated_photoion_cs(
    one_photon: OnePhoton,
    n_qn: int,
    hole_l: int | str,
) -> tuple[ArrFloat64, ArrFloat64]:
    """
    Computes the total (sum over all channels) integrated photoionization cross section for the hole.

    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.

    Returns:
        ekin_eV - an array of electron kinetic energies in eV.
        cs - integrated photoionization cross section for the hole.
    """

    ekin_eV = get_ekin_eV(one_photon, n_qn, hole_l)

    channels = one_photon.get_channels_for_hole(n_qn, hole_l)
    cs = channels.get_cs().sum(axis=0)

    return ekin_eV, cs


def ang_resolved_amplitude_squared_for_angle(
    angle_deg: float,
    M: ArrComplex128_2D,
    hole_l: int | str,
    l_to_index: dict[int | str, int],
) -> ArrFloat64:
    """
    Computes modulus squared of the angularly resolved quantum amplitude for the given emission
    angle. From this amplitude the angularly resolved cross section can be obtained.

    Args:
        angle - angle in degrees.
        M - matrix elements with Coulomb phase.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        l_to_index - a dictionary mapping reachable orbital momenta with the row indices in the data.

    Returns:
        amp_sq - an array with modulus squared of the angularly resolved amplitude.
    """

    # convert angle to radians
    angle_rad = radians(angle_deg)

    # For the modulus squared of the amplitude, create an array of the size of the energy axis (1)
    # in the matrix elements
    amp_sq = np.zeros(M.shape[1], dtype=np.float64)

    # convert to the int format for the following use
    if type(hole_l) is str:
        hole_l = l_to_int(hole_l)

    # reachable final angular momenta from the hole_l after 1 ph
    l_final = final_ls(hole_l, only_reachable=True)
    l_max = np.max(l_final)  # maximum reachable momentum

    # possible m quantum numbers for the ionization;
    # computed using the highest final l (l_max), which automatically includes all lower states
    m_final = np.arange(-l_max, l_max + 1, 1)

    for m in m_final:
        m = int(m)

        l_possible = []  # the list of possible final l, given the m value
        for l in l_final:
            if np.abs(m) <= l:
                l_possible.append(l)

        m_contr = np.zeros(M.shape[1], dtype=np.complex128)

        for l in l_possible:
            pre_factor = (
                sph_harm_y(l, m, angle_rad, 0)
                * ((-1) ** (l - m))
                * np.float64(wigner_3j(l, 1, hole_l, -m, 0, m))
            )
            l_index = l_to_index[l]
            m_contr += pre_factor * M[l_index]

        amp_sq += np.abs(m_contr) ** 2

    # NOTE: mutiple by 2 to account for spin. Is it the valid way??
    amp_sq *= 2

    return amp_sq


def get_ang_resolved_amplitude_squared(
    one_photon: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    angles_deg: list[float],
) -> tuple[ArrFloat64, ArrFloat64_2D]:
    """
    Computes modulus squared of the angularly resolved amplitude at the specified angles.

    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        angles_deg - the list of angles (in degrees).

    Returns:
        ekin_eV - an array of electron kinetic energies in eV.
        amp_sq_ang - angularly resolved photoionization cross section at the specified angles.
    """

    ekin_eV = get_ekin_eV(one_photon, n_qn, hole_l)  # electron kinetic energy
    M = get_matrix_elements_with_coulomb_phase(one_photon, n_qn, hole_l, Z)
    l_to_index = one_photon.get_channels_for_hole(n_qn, hole_l).get_l_to_index(
        out_as="int"
    )

    # create a 2D array to store the result, the first axis corresponds to angles,
    # the second one corresponds to energy values
    N_ang = len(angles_deg)
    N_en = len(ekin_eV)
    amp_sq_ang = np.zeros((N_ang, N_en), dtype=np.float64)

    for i in range(N_ang):
        angle = angles_deg[i]
        amp_sq_ang[i] = ang_resolved_amplitude_squared_for_angle(
            angle, M, hole_l, l_to_index
        )

    return ekin_eV, amp_sq_ang


def get_ang_resolved_photoion_cs(
    one_photon: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    angles_deg: list[float],
) -> tuple[ArrFloat64, ArrFloat64_2D]:
    """
    Computes angularly resolved photoionization cross section for the hole at the specified angles.

    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        angles_deg - the list of angles (in degrees).

    Returns:
        ekin_eV - an array of electron kinetic energies in eV.
        cs_ang - angularly resolved photoionization cross section at the specified angles.
    """

    # get the modulus squared of the angularly resolved amplitude for the given angles
    ekin_eV, amp_sq_ang = get_ang_resolved_amplitude_squared(
        one_photon, n_qn, hole_l, Z, angles_deg
    )

    # prepare qunatities necessary for the cross section calculation
    ekin_hart = get_ekin_Hartree(one_photon, n_qn, hole_l)
    omega_hart = get_omega_Hartree(one_photon, n_qn, hole_l)  # XUV photon energy
    k = wavenumber(ekin_hart)
    C_cs = 2 * np.pi * fine_structure * au_to_Mbarn  # cs conversion factor

    # create a 2D array to store the result, the first axis corresponds to angles,
    # the second one corresponds to energy values
    N_ang = len(angles_deg)
    N_en = len(ekin_eV)
    cs_ang = np.zeros((N_ang, N_en), dtype=np.float64)

    for i in range(N_ang):
        cs_ang[i] = C_cs * amp_sq_ang[i] * k * omega_hart

    return ekin_eV, cs_ang


def get_ang_part_of_photoion_cs(
    one_photon: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    angles_deg: list[float],
) -> tuple[ArrFloat64, ArrFloat64_2D]:
    """
    Computes the angular part of the photoionization cross section at the specified angles.
    The angular part = cs_ang * 4pi / cs_int = 1 + b2 * P2(cos(theta)), where:
    cs_ang is the angularly resolved cross section;
    cs_int is the integrated cross section;
    b2 is the second order real asymmetry parameter;
    P2 is the second order legendre polynomial;

    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        angles_deg - the list of angles (in degrees).

    Returns:
        ekin_eV - an array of electron kinetic energies in eV.
        ang_part - angular part of the photoionization cross section at the specified angles.
    """

    # get the angularly resolved and integrated cross sections
    ekin_eV, cs_ang = get_ang_resolved_photoion_cs(
        one_photon, n_qn, hole_l, Z, angles_deg
    )
    _, cs_int = get_integrated_photoion_cs(one_photon, n_qn, hole_l)

    # a 2D array to store the angular part, the first axis corresponds to angles,
    # the second one corresponds to energy values
    ang_part = np.zeros(cs_ang.shape, dtype=np.float64)

    for i in range(ang_part.shape[0]):
        ang_part[i] = cs_ang[i] * 4 * np.pi / cs_int

    return ekin_eV, ang_part
