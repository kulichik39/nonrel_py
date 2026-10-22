import numpy as np
from fortran_output_analysis.custom_types import (
    ArrFloat64,
    ArrFloat64_2D,
    ArrFloat64_3D,
    ArrComplex128,
    ArrComplex128_2D,
    ArrComplex128_3D,
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
from fortran_output_analysis.twophotons.twophotons import TwoPhotons, final_ls, get_Ks

"""
This namespace contains functions that may be required across different sections of two photon 
analysis (e.g. delays/phases).
NOTE: to retreive the data using the functions below, you have to usually specify which ionisation
pathway you consider: absorption or emission. It can be done through the abs_or_emi parameter.
"""


def get_omega_Hartree(
    two_photons: TwoPhotons, abs_or_emi: str, n_qn: int, hole_l: int | str
) -> ArrFloat64:
    """
    Args:
        two_photons - object of the TwoPhoton class with the data.
        abs_or_emi - ionisation pathway: absorption or emission.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.

    Returns:
        an array of total photon energies (XUV +/- IR) in Hartree for the hole.
    """

    channels = two_photons.get_channels_for_hole(abs_or_emi, n_qn, hole_l)

    return channels.get_omega()


def get_omega_eV(
    two_photons: TwoPhotons, abs_or_emi: str, n_qn: int, hole_l: int | str
) -> ArrFloat64:
    """
    Args:
        two_photons - object of the TwoPhoton class with the data.
        abs_or_emi - ionisation pathway: absorption or emission.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.

    Returns:
        an array of total photon energies (XUV +/- IR) in eV for the hole.
    """

    return get_omega_Hartree(two_photons, abs_or_emi, n_qn, hole_l) * g_eV_per_Hartree


def get_ekin_Hartree(
    two_photons: TwoPhotons, abs_or_emi: str, n_qn: int, hole_l: int | str
) -> ArrFloat64:
    """
    Args:
        two_photons - object of the TwoPhoton class with the data.
        abs_or_emi - ionisation pathway: absorption or emission.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.

    Returns:
        an array of electron kinetic energies (after 2ph) in Hartree for the hole.
    """

    channels = two_photons.get_channels_for_hole(abs_or_emi, n_qn, hole_l)
    hole = channels.hole

    if not hole.binding_energy:
        raise RuntimeError(
            f"The binding energy for the {hole.name} hole is not initialized!"
        )

    return (
        get_omega_Hartree(two_photons, abs_or_emi, n_qn, hole_l) - hole.binding_energy
    )


def get_ekin_eV(
    two_photons: TwoPhotons, abs_or_emi: str, n_qn: int, hole_l: int | str
) -> ArrFloat64:
    """
    Args:
        two_photons - object of the TwoPhoton class with the data.
        abs_or_emi - ionisation pathway: absorption or emission.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.

    Returns:
        an array of electron kinetic energies (after 2ph) in eV for the hole.
    """

    return get_ekin_Hartree(two_photons, abs_or_emi, n_qn, hole_l) * g_eV_per_Hartree


def get_matrix_elements_for_channel(
    two_photons: TwoPhotons,
    abs_or_emi: str,
    n_qn: int,
    hole_l: int | str,
    final_l: int | str,
    K: int,
) -> ArrComplex128:
    """
    Computes matrix elements for the given ionisation channel of the hole.

    Args:
        two_photons - object of the TwoPhoton class with the data.
        abs_or_emi - ionisation pathway: absorption or emission.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        final_l - orbital angular momentum of the final state, in the int or str format.
        K - the rank of the photon interaction.

    Returns:
        an array of matrix elements.
    """

    channels = two_photons.get_channels_for_hole(abs_or_emi, n_qn, hole_l)

    mat_el = channels.get_mat_el_one_channel(final_l, K)  # matrix elements
    non_coul_phase = channels.get_phase_one_channel(final_l, K)  # non-Coulomb phases

    return mat_el * np.exp(1j * non_coul_phase)


def get_matrix_elements(
    two_photons: TwoPhotons,
    abs_or_emi: str,
    n_qn: int,
    hole_l: int | str,
) -> ArrComplex128_3D:
    """
    Computes matrix elements for all ionisation channels of the hole.

    Args:
        two_photons - object of the TwoPhoton class with the data.
        abs_or_emi - ionisation pathway: absorption or emission.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.

    Returns:
        an array of matrix elements.
    """

    channels = two_photons.get_channels_for_hole(abs_or_emi, n_qn, hole_l)
    mat_el = channels.get_mat_el()  # matrix elements
    non_coul_phase = channels.get_phase()  # non-Coulomb phases

    return mat_el * np.exp(1j * non_coul_phase)


def get_coulomb_phase_for_channel(
    two_photons: TwoPhotons,
    abs_or_emi: str,
    n_qn: int,
    hole_l: int | str,
    final_l: int | str,
    Z: int,
) -> ArrFloat64:
    """
    Computes Couloumb phase for the given ionisation channel of the hole.

    Args:
        two_photons - object of the TwoPhoton class with the data.
        abs_or_emi - ionisation pathway: absorption or emission.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        final_l - orbital angular momentum of the final state, in the int or str format.
        Z - charge of the ion.

    Returns:
        an array of Coulomb phase.
    """

    ekin = get_ekin_Hartree(two_photons, abs_or_emi, n_qn, hole_l)

    return coulomb_phase(final_l, ekin, Z)


def get_coulomb_phase(
    two_photons: TwoPhotons, abs_or_emi: str, n_qn: int, hole_l: int | str, Z: int
) -> ArrFloat64_3D:
    """
    Computes Coulomb phase for all ionisation channels of the hole.

    The phase array is 3D:
        1. The first dimension is fixed at 3 and corresponds to the ionisation channels (final_l).
           If a particular ionisation channel is forbidden (l < 0), the whole part is set to 0.
        2. The second dimension is fixed at 3 and corresponds to the rank of the photon interaction K.
           If a particular K is forbidden, the whole part is set to 0. Actually, K doesn't affect
           the Coulomb phase. The only thing that matters is the the ionisation channels (final_l).
        3. The third dimension corresponds to kinetic energies.
    This behavior makes the shape compatible with the matrix elements.

    Args:
        two_photons - object of the TwoPhoton class with the data.
        abs_or_emi - ionisation pathway: absorption or emission.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.

    Returns:
        coulomb_phase_arr - an array of Coulomb phases.
    """

    ekin = get_ekin_Hartree(two_photons, abs_or_emi, n_qn, hole_l)
    N_ekin = len(ekin)

    if type(hole_l) is str:
        hole_l = l_to_int(hole_l)

    N_l = len(
        final_ls(hole_l, only_reachable=False)
    )  # maximum number of reacahable orbital momenta (needed to keep the data dimensionality)

    N_K = len(
        get_Ks(hole_l, 0, only_reachable=False, light_polar="lin")
    )  # maximum number of Ks (needed to keep the data dimensionality)

    # an array to store the coulomb phase
    coulomb_phase_arr = np.zeros((N_l, N_K, N_ekin), dtype=np.float64)

    # get ionisation channels and corresponding data indices
    l_K_to_indices = two_photons.get_channels_for_hole(
        abs_or_emi, n_qn, hole_l
    ).get_l_K_to_indices(out_as="int")

    for final_l, K in l_K_to_indices:
        l_idx, K_idx = l_K_to_indices[(final_l, K)]
        coulomb_phase_arr[l_idx, K_idx] = coulomb_phase(final_l, ekin, Z)

    return coulomb_phase_arr


def get_matrix_elements_with_coulomb_phase_for_channel(
    two_photons: TwoPhotons,
    abs_or_emi: str,
    n_qn: int,
    hole_l: int | str,
    final_l: int | str,
    K: int,
    Z: int,
) -> ArrComplex128:
    """
    Computes matrix elements for the given ionisation channel of the hole and adds Coulomb phase to
    them.

    Args:
        two_photons - object of the TwoPhoton class with the data.
        abs_or_emi - ionisation pathway: absorption or emission.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        final_l - orbital angular momentum of the final state, in the int or str format.
        K - the rank of the photon interaction.
        Z - charge of the ion.

    Returns:
        an array of matrix elements with the Coulomb phase.
    """

    mat_el = get_matrix_elements_for_channel(
        two_photons, abs_or_emi, n_qn, hole_l, final_l, K
    )
    coul_phase = get_coulomb_phase_for_channel(
        two_photons, abs_or_emi, n_qn, hole_l, final_l, Z
    )

    return mat_el * np.exp(1j * coul_phase)


def get_matrix_elements_with_coulomb_phase(
    two_photons: TwoPhotons,
    abs_or_emi: str,
    n_qn: int,
    hole_l: int | str,
    Z: int,
) -> ArrComplex128_3D:
    """
    Computes matrix elements for all ionisation channels of the hole and adds Coulomb phase to
    them.

    Args:
        two_photons - object of the TwoPhoton class with the data.
        abs_or_emi - ionisation pathway: absorption or emission.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.

    Returns:
        an array of matrix elements with the Coulomb phase.
    """

    mat_el = get_matrix_elements(two_photons, abs_or_emi, n_qn, hole_l)
    coul_phase = get_coulomb_phase(two_photons, abs_or_emi, n_qn, hole_l, Z)

    return mat_el * np.exp(1j * coul_phase)


def match_to_sideband_1sim(
    energies_emi: ArrFloat64,
    energies_abs: ArrFloat64,
    M_emi: ArrComplex128_3D,
    M_abs: ArrComplex128_3D,
    g_omega_IR: float,
    energies_mode: str,
) -> tuple[ArrFloat64, ArrComplex128_3D, ArrComplex128_3D]:
    """
    Matches matrix elements to the final sideband energies in the case of one simulation.
    Constructs an array of final sideband energies, and two arrays corresponding to the matched
    matrix elements for the absorption path and the emission path.

    Args:
        energies_emi - energies of the emission path.
        energies_abs - energies of the absorption path.
        M_emi - unmatched emission matrix elements.
        M_abs - unmatched absorption matrix elements.
        g_omega_IR - energy of the IR photon.
        energies_mode - tells which energies we choose for the final sideband, and which energies
                        will be used for the matrix interpolation.

    Returns:
        energies_final - an array of final sideband energies.
        M_emi_matched - an array of matched matrix elements for the emissiom path.
        M_abs_matched - an array of matched matrix elements for the absorption path.
    """

    assert (
        M_emi.shape == M_abs.shape
    ), "Mismatch between the shapes of the emission and absorption matrices from one simulation!"

    energies_final = final_sideband_energies_1sim(
        energies_emi, energies_abs, g_omega_IR, energies_mode
    )

    M_emi_matched = np.zeros(
        (M_emi.shape[0], M_emi.shape[1], len(energies_final)), dtype=np.complex128
    )
    M_abs_matched = np.zeros(
        (M_abs.shape[0], M_abs.shape[1], len(energies_final)), dtype=np.complex128
    )

    for i in range(M_emi.shape[0]):
        for K in range(M_emi.shape[1]):
            M_emi_matched[i, K, :], M_abs_matched[i, K, :] = match_matrix_elements_1sim(
                energies_final,
                energies_emi,
                energies_abs,
                M_emi[i, K, :],
                M_abs[i, K, :],
            )

    return energies_final, M_emi_matched, M_abs_matched


def prepare_matrices_1sim(
    two_photons: TwoPhotons,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    energies_mode: str,
) -> tuple[ArrFloat64, ArrComplex128_3D, ArrComplex128_3D]:
    """
    Works with the case of one simulation. Constructs matrix elements and matches them to the same
    sideband energy.

    Args:
        two_photons - object of the TwoPhoton class with the data.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        energies_mode - tells which energies we choose for the final sideband, and which energies
                        will be used for the matrix interpolation.

    Returns:
        ekin_final_eV - an array of final sideband energies in eV.
        M_emi_matched - an array of matched matrix elements for the emissiom path.
        M_abs_matched - an array of matched matrix elements for the absorption path.
    """

    ekin_eV_emi = get_ekin_eV(two_photons, "emi", n_qn, hole_l)
    ekin_eV_abs = get_ekin_eV(two_photons, "abs", n_qn, hole_l)
    g_omega_IR_eV = two_photons.g_omega_IR_hart * g_eV_per_Hartree

    M_emi = get_matrix_elements_with_coulomb_phase(two_photons, "emi", n_qn, hole_l, Z)
    M_abs = get_matrix_elements_with_coulomb_phase(two_photons, "abs", n_qn, hole_l, Z)

    ekin_final_eV, M_emi_matched, M_abs_matched = match_to_sideband_1sim(
        ekin_eV_emi, ekin_eV_abs, M_emi, M_abs, g_omega_IR_eV, energies_mode
    )

    return ekin_final_eV, M_emi_matched, M_abs_matched


def match_to_sideband_2sim(
    energies_emi: ArrFloat64,
    energies_abs: ArrFloat64,
    M_emi: ArrComplex128_3D,
    M_abs: ArrComplex128_3D,
    energies_mode: str,
    match_mode: str,
) -> tuple[ArrFloat64, ArrComplex128_3D, ArrComplex128_3D]:
    """
    Matches matrix elements to the final sideband energies in the case of two simulations.
    Constructs an array of final sideband energies, and two arrays corresponding to the matched
    matrix elements for the absorption path and the emission path.

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

    assert (M_emi.shape[0] == M_abs.shape[0]) and (
        M_emi.shape[1] == M_abs.shape[1]
    ), "Mismatch between the shapes of the emission and absorption matrices from two simulations!"

    energies_final = final_sideband_energies_2sim(
        energies_emi, energies_abs, energies_mode
    )

    M_emi_matched = np.zeros(
        (M_emi.shape[0], M_emi.shape[1], len(energies_final)),
        dtype=np.complex128,
    )

    M_abs_matched = np.zeros(
        (M_abs.shape[0], M_abs.shape[1], len(energies_final)),
        dtype=np.complex128,
    )

    for i in range(M_emi.shape[0]):
        for K in range(M_emi.shape[1]):
            M_emi_matched[i, K, :], M_abs_matched[i, K, :] = match_matrix_elements_2sim(
                energies_final,
                energies_emi,
                energies_abs,
                M_emi[i, K, :],
                M_abs[i, K, :],
                match_mode,
            )

    return energies_final, M_emi_matched, M_abs_matched


def prepare_matrices_2sim(
    two_photons_emi: TwoPhotons,
    two_photons_abs: TwoPhotons,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    energies_mode: str,
    match_mode: str,
) -> tuple[ArrFloat64, ArrComplex128_3D, ArrComplex128_3D]:
    """
    Works with the case of two simulations. Constructs matrix elements and matches them to the same
    sideband energy.

    Args:
        two_photons_emi - object of the TwoPhotons class corresponding to the emission simulation.
        one_photon_abs - object of the TwoPhotons class corresponding to the absorption simulation.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.
        energies_mode - tells which energies we choose for the final sideband.
        match_mode - the mode of the matrix element matching.

    Returns:
        ekin_final_eV - an array of final sideband energies in eV.
        M_emi_matched - an array of matched matrix elements for the emissiom path.
        M_abs_matched - an array of matched matrix elements for the absorption path.
    """

    ekin_eV_emi = get_ekin_eV(two_photons_emi, "emi", n_qn, hole_l)
    ekin_eV_abs = get_ekin_eV(two_photons_abs, "abs", n_qn, hole_l)

    M_emi = get_matrix_elements_with_coulomb_phase(
        two_photons_emi, "emi", n_qn, hole_l, Z
    )
    M_abs = get_matrix_elements_with_coulomb_phase(
        two_photons_abs, "abs", n_qn, hole_l, Z
    )

    ekin_final_eV, M_emi_matched, M_abs_matched = match_to_sideband_2sim(
        ekin_eV_emi, ekin_eV_abs, M_emi, M_abs, energies_mode, match_mode
    )

    return ekin_final_eV, M_emi_matched, M_abs_matched


def get_prepared_matrices(
    two_photons_1: TwoPhotons,
    n_qn: int,
    hole_l: int | str,
    Z: int,
    energies_mode: str,
    two_photons_2: TwoPhotons | None = None,
    match_mode: str = "interp_both",
) -> tuple[ArrFloat64, ArrComplex128_3D, ArrComplex128_3D]:
    """
    Depending on how many simulations (1 or 2) were provided, constructs emission and absorption
    matrix elements and matches them to the same sideband energy.

    Args:
        two_photons_1 - object of the TwoPhotons class corresponding to the first simulation.
        n_qn - principal quantum number of the hole.
        hole_l - orbital angular momentum of the hole, in the int or str format.
        Z - charge of the ion.

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

        two_photons_2 - second object of the TwoPhotons class if we want to include 2 simulations.
                        In this case, two_photons_1 corresponds to the emission simulation and
                        two_photons_2 to the absorption one.

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

    if two_photons_2:  # if the second simulation is provided
        ekin_final_eV, M_emi_matched, M_abs_matched = prepare_matrices_2sim(
            two_photons_1,
            two_photons_2,
            n_qn,
            hole_l,
            Z,
            energies_mode,
            match_mode,
        )

    else:  # if only one simulation is provided
        ekin_final_eV, M_emi_matched, M_abs_matched = prepare_matrices_1sim(
            two_photons_1, n_qn, hole_l, Z, energies_mode
        )

    return ekin_final_eV, M_emi_matched, M_abs_matched
