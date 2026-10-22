import numpy as np
from math import radians
from scipy.special import sph_harm_y
from sympy.physics.wigner import wigner_3j

from fortran_output_analysis.custom_types import (
    ArrFloat64,
    ArrFloat64_2D,
    ArrComplex128,
    ArrComplex128_2D,
    ArrComplex128_3D,
)
from fortran_output_analysis.common_utility import (
    assert_matrix_shapes,
    l_to_int,
    unwrap_phase_with_nans,
    phase_to_delay,
)
from fortran_output_analysis.twophotons.twophotons import final_ls, get_Ks


def get_integrated_sb_intefer_term(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_3D,
    M_abs_matched: ArrComplex128_3D,
    l_K_to_indices: dict[tuple[int, int], tuple[int, int]],
) -> ArrComplex128:
    """
    Computes integrated sideband interference term. From this term the integrated atomic phase
    can be obtained.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_K_to_indices - a dictionary mapping reachable orbital momenta l and ranks K to the
                         indices in the data arrays. The structure of the dict is as follows:
                                                        {
                                                            (final_l_1, K_1): (row_idx_1, K_idx_1),
                                                            (final_l_2, K_2): (row_idx_2, K_idx_2),
                                                            ...
                                                        }
                          The angular momentum part of the key (final_l) must be in the str format!

    Returns:
        interf_term_int - an array of the integrated interference term.
    """

    assert_matrix_shapes(M_emi_matched, M_abs_matched)

    # For the interference term, create an array of the size of the energy axis (2)
    # in the matrix elements.
    interf_term_int = np.zeros(M_emi_matched.shape[2], dtype=np.complex128)

    # convert to the int format for the following use
    if type(hole_l) is str:
        hole_l = l_to_int(hole_l)

    # reachable final angular momenta from the hole_l after 2 ph
    l_final = final_ls(hole_l, only_reachable=True)

    for l in l_final:  # incoherent sum
        K_possible = get_Ks(hole_l, l, only_reachable=True, light_polar="lin")

        for K in K_possible:  # incoherent sum
            l_index, K_index = l_K_to_indices[(l, K)]
            pre_factor = 1 / (2 * K + 1)
            interf_term_int += (
                pre_factor
                * np.conjugate(M_abs_matched[l_index, K_index])
                * M_emi_matched[l_index, K_index]
            )

    # NOTE: mutiple by 2 to account for spin. Is it the valid way??
    interf_term_int *= 2

    return interf_term_int


def get_integrated_atomic_phase(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_3D,
    M_abs_matched: ArrComplex128_3D,
    l_K_to_indices: dict[tuple[int, int], tuple[int, int]],
    unwrap: bool = True,
) -> ArrFloat64:
    """
    Computes integrated atomic phase.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_K_to_indices - a dictionary mapping reachable orbital momenta l and ranks K to the
                         indices in the data arrays.
        unwrap - tells if the phase should be unwrapped.

    Returns:
        atomic_phase_int - an array of the integrated atomic phase.
    """

    interf_term_int = get_integrated_sb_intefer_term(
        hole_l, M_emi_matched, M_abs_matched, l_K_to_indices
    )

    atomic_phase_int = np.angle(interf_term_int)

    if unwrap:
        atomic_phase_int = unwrap_phase_with_nans(atomic_phase_int)

    return atomic_phase_int


def get_integrated_atomic_delay(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_3D,
    M_abs_matched: ArrComplex128_3D,
    l_K_to_indices: dict[tuple[int, int], tuple[int, int]],
    omega_diff_hart: float,
    unwrap: bool = True,
) -> ArrFloat64:
    """
    Computes integrated atomic delay through the corresponding phase.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_K_to_indices - a dictionary mapping reachable orbital momenta l and ranks K to the
                         indices in the data arrays.
        omega_diff_hart - energy separation (in Hartree) between the absorption and emission paths.
        unwrap - tells if the delay should be unwrapped.

    Returns:
        atomic_delay_int - an array of the integrated atomic delay.
    """

    # get the atomic phase
    atomic_phase_int = get_integrated_atomic_phase(
        hole_l, M_emi_matched, M_abs_matched, l_K_to_indices, unwrap=unwrap
    )

    # convert phase into delay
    atomic_delay_int = phase_to_delay(atomic_phase_int, omega_diff_hart)

    return atomic_delay_int


def ang_resolved_sb_interf_term_for_angle(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_3D,
    M_abs_matched: ArrComplex128_3D,
    l_K_to_indices: dict[tuple[int, int], tuple[int, int]],
    angle_deg: float,
) -> ArrComplex128:
    """
    Computes angularly resolved sideband interference term for the given emission angle.
    From this term the angularly resolved atomic phase can be obtained.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_K_to_indices - a dictionary mapping reachable orbital momenta l and ranks K to the
                         indices in the data arrays.
        angle - angle in degrees.

    Returns:
        interf_term_ang - an array of angularly resolved interference term.
    """

    assert_matrix_shapes(M_emi_matched, M_abs_matched)

    # convert angle to radians
    angle_rad = radians(angle_deg)

    # For the intereference term, create an array of the size of the energy axis (2)
    # in the matrix elements
    interf_term_ang = np.zeros(M_emi_matched.shape[2], dtype=np.complex128)

    # convert to the int format for the following use
    if type(hole_l) is str:
        hole_l = l_to_int(hole_l)

    # reachable final angular momenta from the hole_l after 2 ph
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
        emi_contr = np.zeros(M_emi_matched.shape[2], dtype=np.complex128)
        abs_contr = np.zeros(M_abs_matched.shape[2], dtype=np.complex128)

        for l in l_possible:
            K_possible = get_Ks(hole_l, l, only_reachable=True, light_polar="lin")

            for K in K_possible:
                l_index, K_index = l_K_to_indices[(l, K)]

                pre_factor = (
                    sph_harm_y(l, m, angle_rad, 0)
                    * ((-1) ** (l - m))
                    * np.float64(wigner_3j(l, K, hole_l, -m, 0, m))
                )

                emi_contr += pre_factor * M_emi_matched[l_index, K_index]
                abs_contr += pre_factor * M_abs_matched[l_index, K_index]

        interf_term_ang += np.conjugate(abs_contr) * emi_contr

    # NOTE: mutiple by 2 to account for spin. Is it the valid way??
    interf_term_ang *= 2

    return interf_term_ang


def get_ang_resolved_sb_interf_term(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_3D,
    M_abs_matched: ArrComplex128_3D,
    l_K_to_indices: dict[tuple[int, int], tuple[int, int]],
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
        l_K_to_indices - a dictionary mapping reachable orbital momenta l and ranks K to the
                         indices in the data arrays.
        angles_deg - the list of angles (in degrees).

    Returns:
        interf_term_ang - angularly resolved interference term at the specified angles.
    """

    # create a 2D array to store the result, the first axis corresponds to angles,
    # the second one corresponds to energy values
    N_ang = len(angles_deg)
    interf_term_ang = np.zeros((N_ang, M_emi_matched.shape[2]), dtype=np.complex128)

    for i in range(N_ang):
        angle = angles_deg[i]
        interf_term_ang[i] = ang_resolved_sb_interf_term_for_angle(
            hole_l, M_emi_matched, M_abs_matched, l_K_to_indices, angle
        )
    return interf_term_ang


def get_ang_resolved_atomic_phase(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_3D,
    M_abs_matched: ArrComplex128_3D,
    l_K_to_indices: dict[tuple[int, int], tuple[int, int]],
    angles_deg: list[float],
    unwrap: bool = True,
) -> ArrFloat64_2D:
    """
    Computes angularly resolved atomic phase at the specified angles.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_K_to_indices - a dictionary mapping reachable orbital momenta l and ranks K to the
                         indices in the data arrays.
        angles_deg - the list of angles (in degrees).
        unwrap - tells if the phase should be unwrapped.

    Returns:
        ayomic_phase_ang - angularly resolved atomic phase at the specified angles.
    """

    # get the angularly integrated interference term
    interf_term_ang = get_ang_resolved_sb_interf_term(
        hole_l, M_emi_matched, M_abs_matched, l_K_to_indices, angles_deg
    )

    atomic_phase_ang = np.zeros(interf_term_ang.shape, dtype=np.float64)

    for i in range(interf_term_ang.shape[0]):
        atomic_phase_ang[i] = np.angle(interf_term_ang[i])
        if unwrap:
            atomic_phase_ang[i] = unwrap_phase_with_nans(atomic_phase_ang[i])

    return atomic_phase_ang


def get_ang_resolved_atomic_delay(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_3D,
    M_abs_matched: ArrComplex128_3D,
    l_K_to_indices: dict[tuple[int, int], tuple[int, int]],
    angles_deg: list[float],
    omega_diff_hart: float,
    unwrap: bool = True,
) -> ArrFloat64_2D:
    """
    Computes angularly resolved atomic delay at the specified angles.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_K_to_indices - a dictionary mapping reachable orbital momenta l and ranks K to the
                         indices in the data arrays.
        angles_deg - the list of angles (in degrees).
        omega_diff_hart - energy separation (in Hartree) between the absorption and emission paths.
        unwrap - tells if the delay should be unwrapped.

    Returns:
        atomic_delay_ang - angularly resolved atomic delay at the specified angles.
    """

    # get the angularly resolved phase
    atomic_phase_ang = get_ang_resolved_atomic_phase(
        hole_l,
        M_emi_matched,
        M_abs_matched,
        l_K_to_indices,
        angles_deg,
        unwrap=unwrap,
    )

    atomic_delay_ang = np.zeros(atomic_phase_ang.shape, dtype=np.float64)

    for i in range(atomic_phase_ang.shape[0]):
        atomic_delay_ang[i] = phase_to_delay(atomic_phase_ang[i], omega_diff_hart)

    return atomic_delay_ang


def get_ang_part_of_sb_interf_term(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_3D,
    M_abs_matched: ArrComplex128_3D,
    l_K_to_indices: dict[tuple[int, int], tuple[int, int]],
    angles_deg: list[float],
) -> ArrComplex128_2D:
    """
    Computes the angular part of the sideband interference term at the specified angles.
    The angular part = A(theta) * 4pi / A_int = 1 + b2 * P2(cos(theta)) + b4 * P4(cos(theta)), where:
    A(theta) is the angularly resolved interference term;
    A_int is the integrated interference term;
    b2 and b4 are the second and fourth order complex asymmetry parameters;
    P2 and P4 are the second order legendre polynomials;

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_K_to_indices - a dictionary mapping reachable orbital momenta l and ranks K to the
                         indices in the data arrays.
        angles_deg - the list of angles (in degrees).

    Returns:
        ang_part - angular part of the sideband interference term at the specified angles.
    """

    # get the angularly resolved and integrated interference terms
    interf_term_ang = get_ang_resolved_sb_interf_term(
        hole_l,
        M_emi_matched,
        M_abs_matched,
        l_K_to_indices,
        angles_deg,
    )
    interf_term_int = get_integrated_sb_intefer_term(
        hole_l,
        M_emi_matched,
        M_abs_matched,
        l_K_to_indices,
    )

    # a 2D array to store the angular part, the first axis corresponds to angles,
    # the second one corresponds to energy values
    ang_part = np.zeros(interf_term_ang.shape, dtype=np.complex128)

    for i in range(interf_term_ang.shape[0]):
        ang_part[i] = interf_term_ang[i] * 4 * np.pi / interf_term_int

    return ang_part


def get_ang_part_of_atomic_phase(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_3D,
    M_abs_matched: ArrComplex128_3D,
    l_K_to_indices: dict[tuple[int, int], tuple[int, int]],
    angles_deg: list[float],
    unwrap: bool = True,
) -> ArrFloat64_2D:
    """
    Computes the angular part of the atomic phase at the specified angles.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_K_to_indices - a dictionary mapping reachable orbital momenta l and ranks K to the
                         indices in the data arrays.
        angles_deg - the list of angles (in degrees).
        unwrap - tells if the phase should be unwrapped.

    Returns:
        atomic_phase_ang_part - angular part of the atomic phase at the specified angles.
    """

    # get the angular part of the interf term
    interf_term_ang_part = get_ang_part_of_sb_interf_term(
        hole_l,
        M_emi_matched,
        M_abs_matched,
        l_K_to_indices,
        angles_deg,
    )

    # a 2D array to store the angular part, the first axis corresponds to angles,
    # the second one corresponds to energy values
    atomic_phase_ang_part = np.zeros(interf_term_ang_part.shape, dtype=np.float64)

    for i in range(interf_term_ang_part.shape[0]):
        atomic_phase_ang_part[i] = np.angle(interf_term_ang_part[i])
        if unwrap:
            atomic_phase_ang_part[i] = unwrap_phase_with_nans(atomic_phase_ang_part[i])

    return atomic_phase_ang_part


def get_ang_part_of_atomic_delay(
    hole_l: int | str,
    M_emi_matched: ArrComplex128_3D,
    M_abs_matched: ArrComplex128_3D,
    l_K_to_indices: dict[tuple[int, int], tuple[int, int]],
    angles_deg: list[float],
    omega_diff_hart: float,
    unwrap: bool = True,
) -> ArrFloat64_2D:
    """
    Computes the angular part of the atomic delay at the specified angles.

    Args:
        hole_l - orbital angular momentum of the hole, in the int or str format.
        M_emi_matched - array of matrix elements for the emission path, matched to the final
                        sideband energies.
        M_abs_matched - array of matrix elements for the absortpion path, matched to the final
                        sideband energies.
        l_K_to_indices - a dictionary mapping reachable orbital momenta l and ranks K to the
                         indices in the data arrays.
        angles_deg - the list of angles (in degrees).
        omega_diff_hart - energy separation (in Hartree) between the absorption and emission paths.
        unwrap - tells if the delay should be unwrapped.

    Returns:
        atomic _delay_ang_part - angular part of the atomic delay at the specified angles.
    """

    # get the angular part of the atomic  phase
    atomic_phase_ang_part = get_ang_part_of_atomic_phase(
        hole_l, M_emi_matched, M_abs_matched, l_K_to_indices, angles_deg, unwrap=unwrap
    )

    atomic_delay_ang_part = np.zeros(atomic_phase_ang_part.shape, dtype=np.float64)

    for i in range(atomic_phase_ang_part.shape[0]):
        atomic_delay_ang_part[i] = phase_to_delay(
            atomic_phase_ang_part[i], omega_diff_hart
        )

    return atomic_delay_ang_part
