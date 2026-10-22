import numpy as np
from math import radians
from scipy.special import sph_harm_y
from sympy.physics.wigner import wigner_3j

from fortran_output_analysis.custom_types import (
    ArrFloat64,
    ArrFloat64_2D,
    ArrComplex128,
    ArrComplex128_2D,
)
from fortran_output_analysis.common_utility import (
    assert_matrix_shapes,
    l_to_int,
    unwrap_phase_with_nans,
    phase_to_delay,
)
from fortran_output_analysis.onephoton.onephoton import final_ls


def get_integrated_sb_intefer_term(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_2D,
    M_abs_matched: ArrComplex128_2D,
    l_to_index: dict[int, int],
) -> ArrComplex128:
    """
    Computes integrated sideband interference term. From this term the integrated Wigner phase
    can be obtained.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_to_index - dictionary matching final angular momenta with the row indices in the matrices.
                     The keys (final angular momenta) must be in the int format!

    Returns:
        interf_term_int - an array of the integrated interference term.
    """

    assert_matrix_shapes(M_emi_matched, M_abs_matched)

    # For the interference term, create an array of the size of the energy axis (1)
    # in the matrix elements.
    interf_term_int = np.zeros(M_emi_matched.shape[1], dtype=np.complex128)

    # convert to the int format for the following use
    if type(hole_l) is str:
        hole_l = l_to_int(hole_l)

    # reachable final angular momenta from the hole_l after 1 ph
    l_final = final_ls(hole_l, only_reachable=True)

    for l in l_final:
        l_index = l_to_index[l]
        interf_term_int += np.conjugate(M_abs_matched[l_index]) * M_emi_matched[l_index]

    # for the 1ph integrated term, the sum of the wigner symbols squared over the m-values gives 1/3;
    # also multiply by 2 to account for spin
    interf_term_int *= 2 / 3

    return interf_term_int


def get_integrated_wigner_phase(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_2D,
    M_abs_matched: ArrComplex128_2D,
    l_to_index: dict[int, int],
    unwrap: bool = True,
) -> ArrFloat64:
    """
    Computes integrated wigner phase.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_to_index - dictionary matching final angular momenta with the row indices in the matrices.
                     The keys (final angular momenta) must be in the int format!
        unwrap - tells if the phase should be unwrapped.

    Returns:
        wigner_phase_int - an array of the integrated Wigner phase.
    """

    interf_term_int = get_integrated_sb_intefer_term(
        hole_l, M_emi_matched, M_abs_matched, l_to_index
    )

    wigner_phase_int = np.angle(interf_term_int)

    if unwrap:
        wigner_phase_int = unwrap_phase_with_nans(wigner_phase_int)

    return wigner_phase_int


def get_integrated_wigner_delay(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_2D,
    M_abs_matched: ArrComplex128_2D,
    l_to_index: dict[int, int],
    omega_diff_hart: float,
    unwrap: bool = True,
) -> ArrFloat64:
    """
    Computes integrated wigner delay through the corresponding phase.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_to_index - dictionary matching final angular momenta with the row indices in the matrices.
                     The keys (final angular momenta) must be in the int format!
        omega_diff_hart - energy separation (in Hartree) between the absorption and emission paths.
        unwrap - tells if the delay should be unwrapped.

    Returns:
        wigner_delay_int - an array of the integrated Wigner delay.
    """

    # get the Wigner phase
    wigner_phase_int = get_integrated_wigner_phase(
        hole_l, M_emi_matched, M_abs_matched, l_to_index, unwrap=unwrap
    )

    # convert phase into delay
    wigner_delay_int = phase_to_delay(wigner_phase_int, omega_diff_hart)

    return wigner_delay_int


def ang_resolved_sb_interf_term_for_angle(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_2D,
    M_abs_matched: ArrComplex128_2D,
    l_to_index: dict[int, int],
    angle_deg: float,
) -> ArrComplex128:
    """
    Computes angularly resolved sideband interference term for the given emission angle.
    From this term the angularly resolved wigner phase can be obtained.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_to_index - dictionary matching final angular momenta with the row indices in the matrices.
                     The keys (final angular momenta) must be in the int format!
        angle - angle in degrees.

    Returns:
        interf_term_ang - an array of angularly resolved interference term.
    """

    assert_matrix_shapes(M_emi_matched, M_abs_matched)

    # convert angle to radians
    angle_rad = radians(angle_deg)

    # For the intereference term, create an array of the size of the energy axis (1)
    # in the matrix elements
    interf_term_ang = np.zeros(M_emi_matched.shape[1], dtype=np.complex128)

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

        # arrays for the emission and absorption contributions for the given m value
        emi_contr = np.zeros(M_emi_matched.shape[1], dtype=np.complex128)
        abs_contr = np.zeros(M_abs_matched.shape[1], dtype=np.complex128)

        for l in l_possible:
            pre_factor = (
                sph_harm_y(l, m, angle_rad, 0)
                * ((-1) ** (l - m))
                * np.float64(wigner_3j(l, 1, hole_l, -m, 0, m))
            )
            l_index = l_to_index[l]
            emi_contr += pre_factor * M_emi_matched[l_index]
            abs_contr += pre_factor * M_abs_matched[l_index]

        interf_term_ang += np.conjugate(abs_contr) * emi_contr

    # NOTE: mutiple by 2 to account for spin. Is it the valid way??
    interf_term_ang *= 2

    return interf_term_ang


def get_ang_resolved_sb_interf_term(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_2D,
    M_abs_matched: ArrComplex128_2D,
    l_to_index: dict[int, int],
    angles_deg: list[float],
) -> ArrComplex128_2D:
    """
    Computes angularly resolved sideband inerference term at the specified angles.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_to_index - dictionary matching final angular momenta with the row indices in the matrices.
                     The keys (final angular momenta) must be in the int format!
        angles_deg - the list of angles (in degrees).

    Returns:
        interf_term_ang - angularly resolved interference term at the specified angles.
    """

    # create a 2D array to store the result, the first axis corresponds to angles,
    # the second one corresponds to energy values
    N_ang = len(angles_deg)
    interf_term_ang = np.zeros((N_ang, M_emi_matched.shape[1]), dtype=np.complex128)

    for i in range(N_ang):
        angle = angles_deg[i]
        interf_term_ang[i] = ang_resolved_sb_interf_term_for_angle(
            hole_l, M_emi_matched, M_abs_matched, l_to_index, angle
        )
    return interf_term_ang


def get_ang_resolved_wigner_phase(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_2D,
    M_abs_matched: ArrComplex128_2D,
    l_to_index: dict[int, int],
    angles_deg: list[float],
    unwrap: bool = True,
) -> ArrFloat64_2D:
    """
    Computes angularly resolved wigner phase at the specified angles.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_to_index - dictionary matching final angular momenta with the row indices in the matrices.
                     The keys (final angular momenta) must be in the int format!
        angles_deg - the list of angles (in degrees).
        unwrap - tells if the phase should be unwrapped.

    Returns:
        wigner_phase_ang - angularly resolved Wigner phase at the specified angles.
    """

    # get the angularly integrated interference term
    interf_term_ang = get_ang_resolved_sb_interf_term(
        hole_l, M_emi_matched, M_abs_matched, l_to_index, angles_deg
    )

    wigner_phase_ang = np.zeros(interf_term_ang.shape, dtype=np.float64)

    for i in range(interf_term_ang.shape[0]):
        wigner_phase_ang[i] = np.angle(interf_term_ang[i])
        if unwrap:
            wigner_phase_ang[i] = unwrap_phase_with_nans(wigner_phase_ang[i])

    return wigner_phase_ang


def get_ang_resolved_wigner_delay(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_2D,
    M_abs_matched: ArrComplex128_2D,
    l_to_index: dict[int, int],
    angles_deg: list[float],
    omega_diff_hart: float,
    unwrap: bool = True,
) -> ArrFloat64_2D:
    """
    Computes angularly resolved wigner delay at the specified angles.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_to_index - dictionary matching final angular momenta with the row indices in the matrices.
                     The keys (final angular momenta) must be in the int format!
        angles_deg - the list of angles (in degrees).
        omega_diff_hart - energy separation (in Hartree) between the absorption and emission paths.
        unwrap - tells if the delay should be unwrapped.

    Returns:
        wigner_delay_ang - angularly resolved Wigner delay at the specified angles.
    """

    # get the angularly resolved phase
    wigner_phase_ang = get_ang_resolved_wigner_phase(
        hole_l,
        M_emi_matched,
        M_abs_matched,
        l_to_index,
        angles_deg,
        unwrap=unwrap,
    )

    wigner_delay_ang = np.zeros(wigner_phase_ang.shape, dtype=np.float64)

    for i in range(wigner_phase_ang.shape[0]):
        wigner_delay_ang[i] = phase_to_delay(wigner_phase_ang[i], omega_diff_hart)

    return wigner_delay_ang


def get_ang_part_of_sb_interf_term(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_2D,
    M_abs_matched: ArrComplex128_2D,
    l_to_index: dict[int, int],
    angles_deg: list[float],
) -> ArrComplex128_2D:
    """
    Computes the angular part of the sideband interference term at the specified angles.
    The angular part = A(theta) * 4pi / A_int = 1 + b2 * P2(cos(theta)), where:
    A(theta) is the angularly resolved interference term;
    A_int is the integrated interference term;
    b2 is the second order complex asymmetry parameter;
    P2 is the second order legendre polynomial;

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_to_index - dictionary matching final angular momenta with the row indices in the matrices.
                     The keys (final angular momenta) must be in the int format!
        angles_deg - the list of angles (in degrees).

    Returns:
        ang_part - angular part of the sideband interference term at the specified angles.
    """

    # get the angularly resolved and integrated interference terms
    interf_term_ang = get_ang_resolved_sb_interf_term(
        hole_l,
        M_emi_matched,
        M_abs_matched,
        l_to_index,
        angles_deg,
    )
    interf_term_int = get_integrated_sb_intefer_term(
        hole_l,
        M_emi_matched,
        M_abs_matched,
        l_to_index,
    )

    # a 2D array to store the angular part, the first axis corresponds to angles,
    # the second one corresponds to energy values
    ang_part = np.zeros(interf_term_ang.shape, dtype=np.complex128)

    for i in range(interf_term_ang.shape[0]):
        ang_part[i] = interf_term_ang[i] * 4 * np.pi / interf_term_int

    return ang_part


def get_ang_part_of_wigner_phase(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_2D,
    M_abs_matched: ArrComplex128_2D,
    l_to_index: dict[int, int],
    angles_deg: list[float],
    unwrap: bool = True,
) -> ArrFloat64_2D:
    """
    Computes the angular part of the Wigner phase at the specified angles.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_to_index - dictionary matching final angular momenta with the row indices in the matrices.
                     The keys (final angular momenta) must be in the int format!
        angles_deg - the list of angles (in degrees).
        unwrap - tells if the phase should be unwrapped.

    Returns:
        wigner_phase_ang_part - angular part of the Wigner phase at the specified angles.
    """

    # get the angular part of the inteference term
    interf_term_ang_part = get_ang_part_of_sb_interf_term(
        hole_l,
        M_emi_matched,
        M_abs_matched,
        l_to_index,
        angles_deg,
    )

    # a 2D array to store the angular part, the first axis corresponds to angles,
    # the second one corresponds to energy values
    wigner_phase_ang_part = np.zeros(interf_term_ang_part.shape, dtype=np.float64)

    for i in range(interf_term_ang_part.shape[0]):
        wigner_phase_ang_part[i] = np.angle(interf_term_ang_part[i])
        if unwrap:
            wigner_phase_ang_part[i] = unwrap_phase_with_nans(wigner_phase_ang_part[i])

    return wigner_phase_ang_part


def get_ang_part_of_wigner_delay(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_2D,
    M_abs_matched: ArrComplex128_2D,
    l_to_index: dict[int, int],
    angles_deg: list[float],
    omega_diff_hart: float,
    unwrap: bool = True,
) -> ArrFloat64_2D:
    """
    Computes the angular part of the Wigner delay at the specified angles.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_to_index - dictionary matching final angular momenta with the row indices in the matrices.
                     The keys (final angular momenta) must be in the int format!
        angles_deg - the list of angles (in degrees).
        omega_diff_hart - energy separation (in Hartree) between the absorption and emission paths.
        unwrap - tells if the delay should be unwrapped.

    Returns:
        wigner_delay_ang_part - angular part of the Wigner delay at the specified angles.
    """

    # get the angular part of the Wigner phase
    wigner_phase_ang_part = get_ang_part_of_wigner_phase(
        hole_l, M_emi_matched, M_abs_matched, l_to_index, angles_deg, unwrap=unwrap
    )

    wigner_delay_ang_part = np.zeros(wigner_phase_ang_part.shape, dtype=np.float64)

    for i in range(wigner_phase_ang_part.shape[0]):
        wigner_delay_ang_part[i] = phase_to_delay(
            wigner_phase_ang_part[i], omega_diff_hart
        )

    return wigner_delay_ang_part
