import numpy as np
from fortran_output_analysis.custom_types import (
    ArrFloat64,
    ArrFloat64_2D,
    ArrComplex128,
    ArrComplex128_2D,
)
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree
from fortran_output_analysis.common_utility import (
    l_to_int,
    coulomb_phase,
    final_sideband_energies_1sim,
    match_matrix_elements_1sim,
    final_sideband_energies_2sim,
    match_matrix_elements_2sim,
)
from fortran_output_analysis.onephoton.onephoton import OnePhoton, final_ls

"""
This namespace contains functions that may be required across different sections of one photon 
analysis (e.g. in cross sections, delays/phases).
"""


def get_omega_Hartree(
    one_photon: OnePhoton, n_qn: int, hole_l: int | str
) -> ArrFloat64:
    """
    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.

    Returns:
        an array of XUV photon energies in Hartree for the hole.
    """

    channels = one_photon.get_channels_for_hole(n_qn, hole_l)

    return channels.get_omega()


def get_omega_eV(one_photon: OnePhoton, n_qn: int, hole_l: int | str) -> ArrFloat64:
    """
    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.

    Returns:
        an array of XUV photon energies in eV for the hole.
    """

    return get_omega_Hartree(one_photon, n_qn, hole_l) * g_eV_per_Hartree


def get_ekin_Hartree(one_photon: OnePhoton, n_qn: int, hole_l: int | str) -> ArrFloat64:
    """
    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.

    Returns:
        an array of electron kinetic energies in Hartree for the hole.
    """

    channels = one_photon.get_channels_for_hole(n_qn, hole_l)
    hole = channels.hole

    if not hole.binding_energy:
        raise RuntimeError(
            f"The binding energy for the {hole.name} hole is not initialized!"
        )

    return get_omega_Hartree(one_photon, n_qn, hole_l) - hole.binding_energy


def get_ekin_eV(one_photon: OnePhoton, n_qn: int, hole_l: int | str) -> ArrFloat64:
    """
    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.

    Returns:
        an array of electron kinetic energies in eV for the hole.
    """

    return get_ekin_Hartree(one_photon, n_qn, hole_l) * g_eV_per_Hartree


def get_matrix_elements_for_channel(
    one_photon: OnePhoton, n_qn: int, hole_l: int | str, final_l: int | str
) -> ArrComplex128:
    """
    Computes matrix elements for the given ionisation channel of the hole.

    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        final_l - orbital angular momentum of the final state, in the int or str format.

    Returns:
        an array of matrix elements.
    """

    channels = one_photon.get_channels_for_hole(n_qn, hole_l)

    amp = channels.get_amp_one_channel(final_l)
    phase = channels.get_phase_one_channel(final_l)

    return amp * np.exp(1j * phase)


def get_matrix_elements(
    one_photon: OnePhoton, n_qn: int, hole_l: int | str
) -> ArrComplex128_2D:
    """
    Computes matrix elements for all ionisation channels of the hole.

    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.

    Returns:
        an array of matrix elements.
    """

    channels = one_photon.get_channels_for_hole(n_qn, hole_l)
    amp = channels.get_amp()
    phase = channels.get_phase()

    return amp * np.exp(1j * phase)


def get_coulomb_phase_for_channel(
    one_photon: OnePhoton, n_qn: int, hole_l: int | str, final_l: int | str, Z: int
) -> ArrFloat64:
    """
    Computes Couloumb phase for the given ionisation channel of the hole.

    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        final_l - orbital angular momentum of the final state, in the int or str format.
        Z - charge of the ion.

    Returns:
        an array of Coulomb phase.
    """

    ekin = get_ekin_Hartree(one_photon, n_qn, hole_l)

    return coulomb_phase(final_l, ekin, Z)


def get_coulomb_phase(
    one_photon: OnePhoton, n_qn: int, hole_l: int | str, Z: int
) -> ArrFloat64_2D:
    """
    Computes Coulomb phase for all ionisation channels of the hole.
    The phase array has a fixed number of rows (2) corresponding to the ionisation channels.
    If a particular ionisation channel is forbidden (l < 0), the whole row is set to 0.
    This behavior makes the shape compatible with the matrix elements.

    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.

    Returns:
        coulomb_phase_arr - an array of Coulomb phases.
    """

    ekin = get_ekin_Hartree(one_photon, n_qn, hole_l)
    N_ekin = len(ekin)

    if type(hole_l) is str:
        hole_l = l_to_int(hole_l)

    N_l = len(
        final_ls(hole_l, only_reachable=False)
    )  # maximum number of reacahable orbital momenta (needed to keep the data dimensionality)

    # an array to store the coulomb phase
    coulomb_phase_arr = np.zeros((N_l, N_ekin), dtype=np.float64)

    # get the reachable angular momenta (ionis. channels) and corresponding row indices
    l_to_index = one_photon.get_channels_for_hole(n_qn, hole_l).get_l_to_index(
        out_as="int"
    )

    for final_l in l_to_index:
        idx = l_to_index[final_l]
        coulomb_phase_arr[idx] = coulomb_phase(final_l, ekin, Z)

    return coulomb_phase_arr


def get_matrix_elements_with_coulomb_phase_for_channel(
    one_photon: OnePhoton, n_qn: int, hole_l: int | str, final_l: int | str, Z: int
) -> ArrComplex128:
    """
    Computes matrix elements for the given ionisation channel of the hole and adds Coulomb phase to
    them.

    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        final_l - orbital angular momentum of the final state, in the int or str format.
        Z - charge of the ion.

    Returns:
        an array of matrix elements with the Coulomb phase.
    """

    mat_el = get_matrix_elements_for_channel(one_photon, n_qn, hole_l, final_l)
    coul_phase = get_coulomb_phase_for_channel(one_photon, n_qn, hole_l, final_l, Z)

    return mat_el * np.exp(1j * coul_phase)


def get_matrix_elements_with_coulomb_phase(
    one_photon: OnePhoton, n_qn: int, hole_l: int | str, Z: int
) -> ArrComplex128_2D:
    """
    Computes matrix elements for all ionisation channels of the hole and adds Coulomb phase to
    them.

    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.

    Returns:
        an array of matrix elements with the Coulomb phase.
    """

    mat_el = get_matrix_elements(one_photon, n_qn, hole_l)
    coul_phase = get_coulomb_phase(one_photon, n_qn, hole_l, Z)

    return mat_el * np.exp(1j * coul_phase)


def match_to_sideband_1sim(
    energies_emi: ArrFloat64,
    energies_abs: ArrFloat64,
    M: ArrComplex128_2D,
    g_omega_IR: float,
    energies_mode: str,
) -> tuple[ArrFloat64, ArrComplex128_2D, ArrComplex128_2D]:
    """
    Matches matrix elements to the final sideband energies in the case of one simulation.
    Constructs an array of final sideband energies, and two arrays corresponding to the matched
    matrix elements for the absorption path (lower harmonic) and the emission path (upper harmonic).

    Args:
        energies_emi - energies of the emission path.
        energies_abs - energies of the absorption path.
        M - unmatched matrix elements.
        g_omega_IR - energy of the IR photon.
        energies_mode - tells which energies we choose for the final sideband, and which energies
                        will be used for the matrix interpolation.

    Returns:
        energies_final - an array of final sideband energies.
        M_emi_matched - an array of matched matrix elements for the emissiom path.
        M_abs_matched - an array of matched matrix elements for the absorption path.
    """

    energies_final = final_sideband_energies_1sim(
        energies_emi, energies_abs, g_omega_IR, energies_mode
    )

    M_emi_matched = np.zeros((M.shape[0], len(energies_final)), dtype=np.complex128)
    M_abs_matched = np.zeros((M.shape[0], len(energies_final)), dtype=np.complex128)

    for i in range(M.shape[0]):
        M_emi_matched[i, :], M_abs_matched[i, :] = match_matrix_elements_1sim(
            energies_final, energies_emi, energies_abs, M[i, :], M[i, :]
        )

    return energies_final, M_emi_matched, M_abs_matched


def prepare_matrices_1sim(
    one_photon: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    g_omega_IR_hart: float,
    energies_mode: str,
) -> tuple[ArrFloat64, ArrComplex128_2D, ArrComplex128_2D]:
    """
    Works with the case of one simulation. Constructs matrix elements and matches them to the same
    sideband energy.

    Args:
        one_photon - object of the OnePhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        g_omega_IR_hart - energy of the IR photon in Hartree.
        energies_mode - tells which energies we choose for the final sideband, and which energies
                        will be used for the matrix interpolation.

    Returns:
        ekin_final_eV - an array of final sideband energies in eV.
        M_emi_matched - an array of matched matrix elements for the emissiom path.
        M_abs_matched - an array of matched matrix elements for the absorption path.
    """

    ekin_eV = get_ekin_eV(one_photon, n_qn, hole_l)
    g_omega_IR_eV = g_omega_IR_hart * g_eV_per_Hartree

    # construct sideband (two photon) energies
    energies_emi = ekin_eV - g_omega_IR_eV
    energies_abs = ekin_eV + g_omega_IR_eV

    M = get_matrix_elements_with_coulomb_phase(one_photon, n_qn, hole_l, Z)

    ekin_final_eV, M_emi_matched, M_abs_matched = match_to_sideband_1sim(
        energies_emi, energies_abs, M, g_omega_IR_eV, energies_mode
    )

    return ekin_final_eV, M_emi_matched, M_abs_matched


def match_to_sideband_2sim(
    energies_emi: ArrFloat64,
    energies_abs: ArrFloat64,
    M_emi: ArrComplex128_2D,
    M_abs: ArrComplex128_2D,
    energies_mode: str,
    match_mode: str,
) -> tuple[ArrFloat64, ArrComplex128_2D, ArrComplex128_2D]:
    """
    Matches matrix elements to the final sideband energies in the case of two simulations.
    Constructs an array of final sideband energies, and two arrays corresponding to the matched
    matrix elements for the absorption path (lower harmonic) and the emission path (upper harmonic).

    Args:
        energies_emi - energies of the emission path.
        energies_abs - energies of the absorption path.
        M_emi - unmatched matrix elements for the emission path.
        M_abs - unmatched matrix elements for the absorption path.
        energies_mode - tells which energies we choose for the final sideband.
        match_mode - the mode of the matrix element matching.


    Returns:
        energies_final - an array of final sideband energies.
        M_emi_matched - an array of matched matrix elements for the emissiom path.
        M_abs_matched - an array of matched matrix elements for the absorption path.
    """

    energies_final = final_sideband_energies_2sim(
        energies_emi, energies_abs, energies_mode
    )

    assert (
        M_abs.shape[0] == M_emi.shape[0]
    ), "The number of ionisation channels in absorption and emission matrices are different!"

    M_emi_matched = np.zeros(
        (M_emi.shape[0], len(energies_final)),
        dtype=np.complex128,
    )

    M_abs_matched = np.zeros(
        (M_abs.shape[0], len(energies_final)),
        dtype=np.complex128,
    )

    for i in range(M_abs.shape[0]):
        M_emi_matched[i, :], M_abs_matched[i, :] = match_matrix_elements_2sim(
            energies_final,
            energies_emi,
            energies_abs,
            M_emi[i, :],
            M_abs[i, :],
            match_mode,
        )

    return energies_final, M_emi_matched, M_abs_matched


def prepare_matrices_2sim(
    one_photon_emi: OnePhoton,
    one_photon_abs: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    g_omega_IR_hart_emi: float,
    g_omega_IR_hart_abs: float,
    energies_mode: str,
    match_mode: str,
) -> tuple[ArrFloat64, ArrComplex128_2D, ArrComplex128_2D]:
    """
    Works with the case of two simulations. Constructs matrix elements and matches them to the same
    sideband energy.

    Args:
        one_photon_emi - object of the OnePhoton corresponding to the emission simulation.
        one_photon_abs - object of the OnePhoton corresponding to the absorption simulation.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        g_omega_IR_hart_emi - energy of the IR photon in Hartree for the emission path.
        g_omega_IR_hart_abs - energy of the IR photon in Hartree for the absorption path.
        energies_mode - tells which energies we choose for the final sideband.
        match_mode - the mode of the matrix element matching.

    Returns:
        ekin_final_eV - an array of final sideband energies in eV.
        M_emi_matched - an array of matched matrix elements for the emissiom path.
        M_abs_matched - an array of matched matrix elements for the absorption path.
    """

    ekin_eV_emi = get_ekin_eV(one_photon_emi, n_qn, hole_l)
    ekin_eV_abs = get_ekin_eV(one_photon_abs, n_qn, hole_l)

    g_omega_IR_eV_emi = g_omega_IR_hart_emi * g_eV_per_Hartree
    g_omega_IR_eV_abs = g_omega_IR_hart_abs * g_eV_per_Hartree

    # construct sideband (two photon) energies
    energies_emi = ekin_eV_emi - g_omega_IR_eV_emi
    energies_abs = ekin_eV_abs + g_omega_IR_eV_abs

    M_emi = get_matrix_elements_with_coulomb_phase(one_photon_emi, n_qn, hole_l, Z)
    M_abs = get_matrix_elements_with_coulomb_phase(one_photon_abs, n_qn, hole_l, Z)

    ekin_final_eV, M_emi_matched, M_abs_matched = match_to_sideband_2sim(
        energies_emi, energies_abs, M_emi, M_abs, energies_mode, match_mode
    )

    return ekin_final_eV, M_emi_matched, M_abs_matched


def get_prepared_matrices(
    one_photon_1: OnePhoton,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    g_omega_IR_hart_1: float,
    energies_mode: str,
    one_photon_2: OnePhoton | None = None,
    g_omega_IR_hart_2: float | None = None,
    match_mode: str = "interp_both",
) -> tuple[ArrFloat64, ArrComplex128_2D, ArrComplex128_2D]:
    """
    Depending on how many simulations (1 or 2) were provided, constructs emission and absorption
    matrix elements and matches them to the same sideband energy.

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
        M_emi_matched - an array of matched matrix elements for the emissiom path.
        M_abs_matched - an array of matched matrix elements for the absorption path.
    """

    if one_photon_2:  # if the second simulation is provided
        assert (
            g_omega_IR_hart_2
        ), "IR photon energy for the second simulation is not provided!"

        ekin_final_eV, M_emi_matched, M_abs_matched = prepare_matrices_2sim(
            one_photon_1,
            one_photon_2,
            n_qn,
            hole_l,
            Z,
            g_omega_IR_hart_1,
            g_omega_IR_hart_2,
            energies_mode,
            match_mode,
        )
    else:  # if only one simulation is provided
        ekin_final_eV, M_emi_matched, M_abs_matched = prepare_matrices_1sim(
            one_photon_1, n_qn, hole_l, Z, g_omega_IR_hart_1, energies_mode
        )

    return ekin_final_eV, M_emi_matched, M_abs_matched
