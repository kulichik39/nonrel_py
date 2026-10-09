import numpy as np
from math import radians
from scipy.special import sph_harm_y
from sympy.physics.wigner import wigner_3j

from fortran_output_analysis.constants_and_parameters import fine_structure, au_to_Mbarn
from fortran_output_analysis.types import (
    ArrFloat64,
    ArrFloat64_2D,
    ArrComplex128,
    ArrComplex128_2D,
)
from fortran_output_analysis.common_utility import (
    l_to_int,
    unwrap_phase_with_nans,
    compute_omega_diff,
    phase_to_delay,
)
from fortran_output_analysis.common_utility import wavenumber
from fortran_output_analysis.onephoton.onephoton import OnePhoton, final_ls
from fortran_output_analysis.onephoton.utilities import (
    get_ekin_eV,
    get_ekin_Hartree,
    get_omega_Hartree,
    get_matrix_elements_with_coulomb_phase,
    get_prepared_matrices,
)


def get_integrated_sb_intefer_term(
    one_photon_1: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    g_omega_IR_hart_1: float,
    energies_mode: str,
    one_photon_2: OnePhoton | None = None,
    g_omega_IR_hart_2: float | None = None,
    match_mode: str = "interp_both",
) -> tuple[ArrFloat64, ArrComplex128]:
    """
    Computes integrated sideband interference term. From this term the integrated Wigner phase
    can be obtained.

    Args:
        one_photon_1 - object of the OnePhoton class corresponding to the first simulation.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        g_omega_IR_hart_1 - energy of the IR photon in Hartree in the first simulation.

        energies_mode - tells which energies we choose for the final sideband.
                        Possible options differ between the case of one simulation and the case
                        of two simulations.

                        Possible options, 1 simulation:
                        "emi" - emission energies starting from the first available absorption point.
                        "abs" - absorption energies cut at the last available emission point.

                        Possible options, 2 simulations:
                        "emi" - emission energies.
                        "abs" - absorption energies.
                        "both" - combined emission and absorption energies.

        one_photon_2 - second object of the OnePhoton class if we want to include 2 simulations.
                       In this case, one_photon_1 correspond to the emission simulation and
                       one_photon_2 to the absorption one.
        g_omega_IR_hart_2 - energy of the IR photon in Hartree in the second simulation, if included.
        match_mode - Required for 2 simulations only. Specifies the mode of the matrix element
                     matching.
                     Possible options:
                     "interp_both" - interpolate both absorption and emission paths for the final
                                     energies.
                     "lin_extrap_emi" - linearly extrapolate the emission path and interpolate the
                                        absorption path for the final energies.
                     "lin_extrap_emi_left" - linearly extrapolate the emission path to the left using
                                             the first two points and interpolate the absosrption
                                             path for the final energies.

    Returns:
        ekin_final_eV - an array of final sideband energies in eV.
        interf_term_int - an array with the integrated interference term.
    """

    # get the final sideband energies with the matched matrices
    ekin_final_eV, M_emi_matched, M_abs_matched = get_prepared_matrices(
        one_photon_1,
        n_qn,
        hole_l,
        Z,
        g_omega_IR_hart_1,
        energies_mode,
        one_photon_2=one_photon_2,
        g_omega_IR_hart_2=g_omega_IR_hart_2,
        match_mode=match_mode,
    )

    # the dictionary matching final angular momenta with the row indices in the matrices
    l_to_index = one_photon_1.get_channels_for_hole(n_qn, hole_l).get_l_to_index(
        out_as="int"
    )

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

    return ekin_final_eV, interf_term_int


def get_integrated_wigner_phase(
    one_photon_1: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    g_omega_IR_hart_1: float,
    energies_mode: str,
    one_photon_2: OnePhoton | None = None,
    g_omega_IR_hart_2: float | None = None,
    match_mode: str = "interp_both",
    unwrap: bool = True,
) -> tuple[ArrFloat64, ArrFloat64]:
    """
    Computes integrated wigner phase. Can compute for one or two simulations. If 2 simulations
    are provided, then the first one_photon object should correspond to the emission path,
    while the second one to the absorption path.

    Args:
        one_photon_1 - object of the OnePhoton class corresponding to the first simulation.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        g_omega_IR_hart_1 - energy of the IR photon in Hartree in the first simulation.

        energies_mode - tells which energies we choose for the final sideband.
                        Possible options differ between the case of one simulation and the case
                        of two simulations.

                        Possible options, 1 simulation:
                        "emi" - emission energies starting from the first available absorption point.
                        "abs" - absorption energies cut at the last available emission point.

                        Possible options, 2 simulations:
                        "emi" - emission energies.
                        "abs" - absorption energies.
                        "both" - combined emission and absorption energies.

        one_photon_2 - second object of the OnePhoton class if we want to include 2 simulations.
                       In this case, one_photon_1 correspond to the emission simulation and
                       one_photon_2 to the absorption one.
        g_omega_IR_hart_2 - energy of the IR photon in Hartree in the second simulation, if included.
        match_mode - Required for 2 simulations only. Specifies the mode of the matrix element
                     matching.
                     Possible options:
                     "interp_both" - interpolate both absorption and emission paths for the final
                                     energies.
                     "lin_extrap_emi" - linearly extrapolate the emission path and interpolate the
                                        absorption path for the final energies.
                     "lin_extrap_emi_left" - linearly extrapolate the emission path to the left using
                                             the first two points and interpolate the absosrption
                                             path for the final energies.

        unwrap - tells if the phase should be unwrapped.

    Returns:
        ekin_final_eV - an array of final sideband energies in eV.
        wigner_phase_int - an array with the integrated Wigner phase.
    """

    ekin_final_eV, interf_term_int = get_integrated_sb_intefer_term(
        one_photon_1,
        n_qn,
        hole_l,
        Z,
        g_omega_IR_hart_1,
        energies_mode,
        one_photon_2=one_photon_2,
        g_omega_IR_hart_2=g_omega_IR_hart_2,
        match_mode=match_mode,
    )

    wigner_phase_int = np.angle(interf_term_int)

    if unwrap:
        wigner_phase_int = unwrap_phase_with_nans(wigner_phase_int)

    return ekin_final_eV, wigner_phase_int


def get_integrated_wigner_delay(
    one_photon_1: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    g_omega_IR_hart_1: float,
    energies_mode: str,
    one_photon_2: OnePhoton | None = None,
    g_omega_IR_hart_2: float | None = None,
    match_mode: str = "interp_both",
    unwrap: bool = True,
) -> tuple[ArrFloat64, ArrFloat64]:
    """
    Computes integrated wigner delay through the corresponding phase.

    Args:
        one_photon_1 - object of the OnePhoton class corresponding to the first simulation.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        g_omega_IR_hart_1 - energy of the IR photon in Hartree in the first simulation.

        energies_mode - tells which energies we choose for the final sideband.
                        Possible options differ between the case of one simulation and the case
                        of two simulations.

                        Possible options, 1 simulation:
                        "emi" - emission energies starting from the first available absorption point.
                        "abs" - absorption energies cut at the last available emission point.

                        Possible options, 2 simulations:
                        "emi" - emission energies.
                        "abs" - absorption energies.
                        "both" - combined emission and absorption energies.

        one_photon_2 - second object of the OnePhoton class if we want to include 2 simulations.
                       In this case, one_photon_1 correspond to the emission simulation and
                       one_photon_2 to the absorption one.
        g_omega_IR_hart_2 - energy of the IR photon in Hartree in the second simulation, if included.
        match_mode - Required for 2 simulations only. Specifies the mode of the matrix element
                     matching.
                     Possible options:
                     "interp_both" - interpolate both absorption and emission paths for the final
                                     energies.
                     "lin_extrap_emi" - linearly extrapolate the emission path and interpolate the
                                        absorption path for the final energies.
                     "lin_extrap_emi_left" - linearly extrapolate the emission path to the left using
                                             the first two points and interpolate the absosrption
                                             path for the final energies.

        unwrap - tells if the delay should be unwrapped.

    Returns:
        ekin_final_eV - an array of final sideband energies in eV.
        wigner_delay_int - an array with the integrated Wigner delay.
    """

    # get the Wigner phase
    ekin_final_eV, wigner_phase_int = get_integrated_wigner_phase(
        one_photon_1,
        n_qn,
        hole_l,
        Z,
        g_omega_IR_hart_1,
        energies_mode,
        one_photon_2=one_photon_2,
        g_omega_IR_hart_2=g_omega_IR_hart_2,
        match_mode=match_mode,
        unwrap=unwrap,
    )

    # energy separation (in Hartree) between the absorption and emission paths
    omega_diff_hart = compute_omega_diff(
        g_omega_IR_hart_1, g_omega_IR_2=g_omega_IR_hart_2
    )

    # convert phase into delay
    wigner_delay_int = phase_to_delay(wigner_phase_int, omega_diff_hart)

    return ekin_final_eV, wigner_delay_int


def ang_resolved_sb_interf_term_for_angle(
    angle_deg: float,
    M_emi_matched: ArrComplex128_2D,
    M_abs_matched: ArrComplex128_2D,
    hole_l: int | str,
    l_to_index: dict[int | str, int],
) -> ArrComplex128:
    """
    Computes angularly resolved sideband interference term for the given emission angle.
    From this term the angularly resolved wigner phase can be obtained.

    Args:
        angle - angle in degrees.
        M_emi_matched - emission matrix elements matched to the final sideband energy.
        M_abs_matched - absorption matrix elements matched to the final sideband energy.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        l_to_index - a dictionary mapping reachable orbital momenta with the row indices in the data.

    Returns:
        interf_term_ang - an array with angularly resolved interference term.
    """

    assert (
        M_emi_matched.shape == M_abs_matched.shape
    ), "The shapes of the input matrices must be the same!"

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
    one_photon_1: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    g_omega_IR_hart_1: float,
    energies_mode: str,
    angles_deg: list[float],
    one_photon_2: OnePhoton | None = None,
    g_omega_IR_hart_2: float | None = None,
    match_mode: str = "interp_both",
) -> tuple[ArrFloat64, ArrComplex128_2D]:
    """
    Computes angularly resolved sideband inerference term at the specified angles.

    Args:
        one_photon_1 - object of the OnePhoton class corresponding to the first simulation.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        g_omega_IR_hart_1 - energy of the IR photon in Hartree in the first simulation.

        energies_mode - tells which energies we choose for the final sideband.
                        Possible options differ between the case of one simulation and the case
                        of two simulations.

                        Possible options, 1 simulation:
                        "emi" - emission energies starting from the first available absorption point.
                        "abs" - absorption energies cut at the last available emission point.

                        Possible options, 2 simulations:
                        "emi" - emission energies.
                        "abs" - absorption energies.
                        "both" - combined emission and absorption energies.

        angles_deg - the list of angles (in degrees).

        one_photon_2 - second object of the OnePhoton class if we want to include 2 simulations.
                       In this case, one_photon_1 correspond to the emission simulation and
                       one_photon_2 to the absorption one.
        g_omega_IR_hart_2 - energy of the IR photon in Hartree in the second simulation, if included.
        match_mode - Required for 2 simulations only. Specifies the mode of the matrix element
                     matching.
                     Possible options:
                     "interp_both" - interpolate both absorption and emission paths for the final
                                     energies.
                     "lin_extrap_emi" - linearly extrapolate the emission path and interpolate the
                                        absorption path for the final energies.
                     "lin_extrap_emi_left" - linearly extrapolate the emission path to the left using
                                             the first two points and interpolate the absosrption
                                             path for the final energies.

    Returns:
        ekin_final_eV - an array of final sideband energies in eV.
        interf_term_ang - angularly resolved interference term at the specified angles.
    """

    # get the final sideband energies with the matched matrices
    ekin_final_eV, M_emi_matched, M_abs_matched = get_prepared_matrices(
        one_photon_1,
        n_qn,
        hole_l,
        Z,
        g_omega_IR_hart_1,
        energies_mode,
        one_photon_2=one_photon_2,
        g_omega_IR_hart_2=g_omega_IR_hart_2,
        match_mode=match_mode,
    )

    # the dictionary matching final angular momenta with the row indices in the matrices
    l_to_index = one_photon_1.get_channels_for_hole(n_qn, hole_l).get_l_to_index(
        out_as="int"
    )

    # create a 2D array to store the result, the first axis corresponds to angles,
    # the second one corresponds to energy values
    N_ang = len(angles_deg)
    N_en = len(ekin_final_eV)
    interf_term_ang = np.zeros((N_ang, N_en), dtype=np.complex128)

    for i in range(N_ang):
        angle = angles_deg[i]
        interf_term_ang[i] = ang_resolved_sb_interf_term_for_angle(
            angle, M_emi_matched, M_abs_matched, hole_l, l_to_index
        )
    return ekin_final_eV, interf_term_ang


def get_ang_resolved_wigner_phase(
    one_photon_1: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    g_omega_IR_hart_1: float,
    energies_mode: str,
    angles_deg: list[float],
    one_photon_2: OnePhoton | None = None,
    g_omega_IR_hart_2: float | None = None,
    match_mode: str = "interp_both",
    unwrap: bool = True,
) -> tuple[ArrFloat64, ArrFloat64_2D]:
    """
    Computes angularly resolved wigner phase at the specified angles.

    Args:
        one_photon_1 - object of the OnePhoton class corresponding to the first simulation.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        g_omega_IR_hart_1 - energy of the IR photon in Hartree in the first simulation.

        energies_mode - tells which energies we choose for the final sideband.
                        Possible options differ between the case of one simulation and the case
                        of two simulations.

                        Possible options, 1 simulation:
                        "emi" - emission energies starting from the first available absorption point.
                        "abs" - absorption energies cut at the last available emission point.

                        Possible options, 2 simulations:
                        "emi" - emission energies.
                        "abs" - absorption energies.
                        "both" - combined emission and absorption energies.

        angles_deg - the list of angles (in degrees).

        one_photon_2 - second object of the OnePhoton class if we want to include 2 simulations.
                       In this case, one_photon_1 correspond to the emission simulation and
                       one_photon_2 to the absorption one.
        g_omega_IR_hart_2 - energy of the IR photon in Hartree in the second simulation, if included.
        match_mode - Required for 2 simulations only. Specifies the mode of the matrix element
                     matching.
                     Possible options:
                     "interp_both" - interpolate both absorption and emission paths for the final
                                     energies.
                     "lin_extrap_emi" - linearly extrapolate the emission path and interpolate the
                                        absorption path for the final energies.
                     "lin_extrap_emi_left" - linearly extrapolate the emission path to the left using
                                             the first two points and interpolate the absosrption
                                             path for the final energies.
        unwrap - tells if the phase should be unwrapped.

    Returns:
        ekin_final_eV - an array of final sideband energies in eV.
        wigner_phase_ang - angularly resolved Wigner phase at the specified angles.
    """

    # get the angularly integrated interference term
    ekin_final_eV, interf_term_ang = get_ang_resolved_sb_interf_term(
        one_photon_1,
        n_qn,
        hole_l,
        Z,
        g_omega_IR_hart_1,
        energies_mode,
        angles_deg,
        one_photon_2=one_photon_2,
        g_omega_IR_hart_2=g_omega_IR_hart_2,
        match_mode=match_mode,
    )

    wigner_phase_ang = np.zeros(interf_term_ang.shape, dtype=np.float64)

    for i in range(interf_term_ang.shape[0]):
        wigner_phase_ang[i] = np.angle(interf_term_ang[i])
        if unwrap:
            wigner_phase_ang[i] = unwrap_phase_with_nans(wigner_phase_ang[i])

    return ekin_final_eV, wigner_phase_ang


def get_ang_resolved_wigner_delay(
    one_photon_1: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    g_omega_IR_hart_1: float,
    energies_mode: str,
    angles_deg: list[float],
    one_photon_2: OnePhoton | None = None,
    g_omega_IR_hart_2: float | None = None,
    match_mode: str = "interp_both",
    unwrap: bool = True,
) -> tuple[ArrFloat64, ArrFloat64_2D]:
    """
    Computes angularly resolved wigner delay at the specified angles.

    Args:
        one_photon_1 - object of the OnePhoton class corresponding to the first simulation.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        g_omega_IR_hart_1 - energy of the IR photon in Hartree in the first simulation.

        energies_mode - tells which energies we choose for the final sideband.
                        Possible options differ between the case of one simulation and the case
                        of two simulations.

                        Possible options, 1 simulation:
                        "emi" - emission energies starting from the first available absorption point.
                        "abs" - absorption energies cut at the last available emission point.

                        Possible options, 2 simulations:
                        "emi" - emission energies.
                        "abs" - absorption energies.
                        "both" - combined emission and absorption energies.

        angles_deg - the list of angles (in degrees).

        one_photon_2 - second object of the OnePhoton class if we want to include 2 simulations.
                       In this case, one_photon_1 correspond to the emission simulation and
                       one_photon_2 to the absorption one.
        g_omega_IR_hart_2 - energy of the IR photon in Hartree in the second simulation, if included.
        match_mode - Required for 2 simulations only. Specifies the mode of the matrix element
                     matching.
                     Possible options:
                     "interp_both" - interpolate both absorption and emission paths for the final
                                     energies.
                     "lin_extrap_emi" - linearly extrapolate the emission path and interpolate the
                                        absorption path for the final energies.
                     "lin_extrap_emi_left" - linearly extrapolate the emission path to the left using
                                             the first two points and interpolate the absosrption
                                             path for the final energies.
        unwrap - tells if the delay should be unwrapped.

    Returns:
        ekin_final_eV - an array of final sideband energies in eV.
        wigner_delay_ang - angularly resolved Wigner delay at the specified angles.
    """

    # get the angularly integrated interference term
    ekin_final_eV, wigner_phase_ang = get_ang_resolved_wigner_phase(
        one_photon_1,
        n_qn,
        hole_l,
        Z,
        g_omega_IR_hart_1,
        energies_mode,
        angles_deg,
        one_photon_2=one_photon_2,
        g_omega_IR_hart_2=g_omega_IR_hart_2,
        match_mode=match_mode,
        unwrap=unwrap,
    )

    wigner_delay_ang = np.zeros(wigner_phase_ang.shape, dtype=np.float64)

    omega_diff_hart = compute_omega_diff(
        g_omega_IR_hart_1, g_omega_IR_2=g_omega_IR_hart_2
    )

    for i in range(wigner_phase_ang.shape[0]):
        wigner_delay_ang[i] = phase_to_delay(wigner_phase_ang[i], omega_diff_hart)

    return ekin_final_eV, wigner_delay_ang


def get_ang_part_of_sb_interf_term(
    one_photon_1: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    g_omega_IR_hart_1: float,
    energies_mode: str,
    angles_deg: list[float],
    one_photon_2: OnePhoton | None = None,
    g_omega_IR_hart_2: float | None = None,
    match_mode: str = "interp_both",
) -> tuple[ArrFloat64, ArrComplex128_2D]:
    """
    Computes the angular part of the sideband interference term at the specified angles.
    The angular part = A(theta) * 4pi / A_int = 1 + b2 * P2(cos(theta)), where:
    A(theta) is the angularly resolved interference term;
    A_int is the integrated interference term;
    b2 is the second order complex asymmetry parameter;
    P2 is the second order legendre polynomial;

    Args:
        one_photon_1 - object of the OnePhoton class corresponding to the first simulation.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        g_omega_IR_hart_1 - energy of the IR photon in Hartree in the first simulation.

        energies_mode - tells which energies we choose for the final sideband.
                        Possible options differ between the case of one simulation and the case
                        of two simulations.

                        Possible options, 1 simulation:
                        "emi" - emission energies starting from the first available absorption point.
                        "abs" - absorption energies cut at the last available emission point.

                        Possible options, 2 simulations:
                        "emi" - emission energies.
                        "abs" - absorption energies.
                        "both" - combined emission and absorption energies.

        angles_deg - the list of angles (in degrees).

        one_photon_2 - second object of the OnePhoton class if we want to include 2 simulations.
                       In this case, one_photon_1 correspond to the emission simulation and
                       one_photon_2 to the absorption one.
        g_omega_IR_hart_2 - energy of the IR photon in Hartree in the second simulation, if included.
        match_mode - Required for 2 simulations only. Specifies the mode of the matrix element
                     matching.
                     Possible options:
                     "interp_both" - interpolate both absorption and emission paths for the final
                                     energies.
                     "lin_extrap_emi" - linearly extrapolate the emission path and interpolate the
                                        absorption path for the final energies.
                     "lin_extrap_emi_left" - linearly extrapolate the emission path to the left using
                                             the first two points and interpolate the absosrption
                                             path for the final energies.
    Returns:
        ekin_final_eV - an array of electron kinetic energies in eV.
        ang_part - angular part of the sideband interference term at the specified angles.
    """

    # get the angularly resolved and integrated interference terms
    ekin_final_eV, interf_term_ang = get_ang_resolved_sb_interf_term(
        one_photon_1,
        n_qn,
        hole_l,
        Z,
        g_omega_IR_hart_1,
        energies_mode,
        angles_deg,
        one_photon_2=one_photon_2,
        g_omega_IR_hart_2=g_omega_IR_hart_2,
        match_mode=match_mode,
    )
    _, interf_term_int = get_integrated_sb_intefer_term(
        one_photon_1,
        n_qn,
        hole_l,
        Z,
        g_omega_IR_hart_1,
        energies_mode,
        one_photon_2=one_photon_2,
        g_omega_IR_hart_2=g_omega_IR_hart_2,
        match_mode=match_mode,
    )

    # a 2D array to store the angular part, the first axis corresponds to angles,
    # the second one corresponds to energy values
    ang_part = np.zeros(interf_term_ang.shape, dtype=np.complex128)

    for i in range(interf_term_ang.shape[0]):
        ang_part[i] = interf_term_ang[i] * 4 * np.pi / interf_term_int

    return ekin_final_eV, ang_part


def get_ang_part_of_wigner_phase(
    one_photon_1: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    g_omega_IR_hart_1: float,
    energies_mode: str,
    angles_deg: list[float],
    one_photon_2: OnePhoton | None = None,
    g_omega_IR_hart_2: float | None = None,
    match_mode: str = "interp_both",
    unwrap: bool = True,
) -> tuple[ArrFloat64, ArrFloat64_2D]:
    """
    Computes the angular part of the Wigner phase at the specified angles.

    Args:
        one_photon_1 - object of the OnePhoton class corresponding to the first simulation.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        g_omega_IR_hart_1 - energy of the IR photon in Hartree in the first simulation.

        energies_mode - tells which energies we choose for the final sideband.
                        Possible options differ between the case of one simulation and the case
                        of two simulations.

                        Possible options, 1 simulation:
                        "emi" - emission energies starting from the first available absorption point.
                        "abs" - absorption energies cut at the last available emission point.

                        Possible options, 2 simulations:
                        "emi" - emission energies.
                        "abs" - absorption energies.
                        "both" - combined emission and absorption energies.

        angles_deg - the list of angles (in degrees).

        one_photon_2 - second object of the OnePhoton class if we want to include 2 simulations.
                       In this case, one_photon_1 correspond to the emission simulation and
                       one_photon_2 to the absorption one.
        g_omega_IR_hart_2 - energy of the IR photon in Hartree in the second simulation, if included.
        match_mode - Required for 2 simulations only. Specifies the mode of the matrix element
                     matching.
                     Possible options:
                     "interp_both" - interpolate both absorption and emission paths for the final
                                     energies.
                     "lin_extrap_emi" - linearly extrapolate the emission path and interpolate the
                                        absorption path for the final energies.
                     "lin_extrap_emi_left" - linearly extrapolate the emission path to the left using
                                             the first two points and interpolate the absosrption
                                             path for the final energies.
        unwrap - tells if the phase should be unwrapped.

    Returns:
        ekin_final_eV - an array of electron kinetic energies in eV.
        wigner_phase_ang_part - angular part of the Wigner phase at the specified angles.
    """

    # get the angular part of
    ekin_final_eV, interf_term_ang_part = get_ang_part_of_sb_interf_term(
        one_photon_1,
        n_qn,
        hole_l,
        Z,
        g_omega_IR_hart_1,
        energies_mode,
        angles_deg,
        one_photon_2=one_photon_2,
        g_omega_IR_hart_2=g_omega_IR_hart_2,
        match_mode=match_mode,
    )

    # a 2D array to store the angular part, the first axis corresponds to angles,
    # the second one corresponds to energy values
    wigner_phase_ang_part = np.zeros(interf_term_ang_part.shape, dtype=np.float64)

    for i in range(interf_term_ang_part.shape[0]):
        wigner_phase_ang_part[i] = np.angle(interf_term_ang_part[i])
        if unwrap:
            wigner_phase_ang_part[i] = unwrap_phase_with_nans(wigner_phase_ang_part[i])

    return ekin_final_eV, wigner_phase_ang_part


def get_ang_part_of_wigner_delay(
    one_photon_1: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    g_omega_IR_hart_1: float,
    energies_mode: str,
    angles_deg: list[float],
    one_photon_2: OnePhoton | None = None,
    g_omega_IR_hart_2: float | None = None,
    match_mode: str = "interp_both",
    unwrap: bool = True,
) -> tuple[ArrFloat64, ArrFloat64_2D]:
    """
    Computes the angular part of the Wigner delay at the specified angles.

    Args:
        one_photon_1 - object of the OnePhoton class corresponding to the first simulation.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        g_omega_IR_hart_1 - energy of the IR photon in Hartree in the first simulation.

        energies_mode - tells which energies we choose for the final sideband.
                        Possible options differ between the case of one simulation and the case
                        of two simulations.

                        Possible options, 1 simulation:
                        "emi" - emission energies starting from the first available absorption point.
                        "abs" - absorption energies cut at the last available emission point.

                        Possible options, 2 simulations:
                        "emi" - emission energies.
                        "abs" - absorption energies.
                        "both" - combined emission and absorption energies.

        angles_deg - the list of angles (in degrees).

        one_photon_2 - second object of the OnePhoton class if we want to include 2 simulations.
                       In this case, one_photon_1 correspond to the emission simulation and
                       one_photon_2 to the absorption one.
        g_omega_IR_hart_2 - energy of the IR photon in Hartree in the second simulation, if included.
        match_mode - Required for 2 simulations only. Specifies the mode of the matrix element
                     matching.
                     Possible options:
                     "interp_both" - interpolate both absorption and emission paths for the final
                                     energies.
                     "lin_extrap_emi" - linearly extrapolate the emission path and interpolate the
                                        absorption path for the final energies.
                     "lin_extrap_emi_left" - linearly extrapolate the emission path to the left using
                                             the first two points and interpolate the absosrption
                                             path for the final energies.
        unwrap - tells if the phase should be unwrapped.

    Returns:
        ekin_final_eV - an array of electron kinetic energies in eV.
        wigner_delay_ang_part - angular part of the Wigner delay at the specified angles.
    """

    # get the angular part of the Wigner phase
    ekin_final_eV, wigner_phase_ang_part = get_ang_part_of_wigner_phase(
        one_photon_1,
        n_qn,
        hole_l,
        Z,
        g_omega_IR_hart_1,
        energies_mode,
        angles_deg,
        one_photon_2=one_photon_2,
        g_omega_IR_hart_2=g_omega_IR_hart_2,
        match_mode=match_mode,
        unwrap=unwrap,
    )

    wigner_delay_ang_part = np.zeros(wigner_phase_ang_part.shape, dtype=np.float64)

    omega_diff_hart = compute_omega_diff(
        g_omega_IR_hart_1, g_omega_IR_2=g_omega_IR_hart_2
    )

    for i in range(wigner_phase_ang_part.shape[0]):
        wigner_delay_ang_part[i] = phase_to_delay(
            wigner_phase_ang_part[i], omega_diff_hart
        )

    return ekin_final_eV, wigner_delay_ang_part
