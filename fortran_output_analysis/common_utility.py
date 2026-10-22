import numpy as np
from scipy.special import gamma
import json
from scipy.interpolate import InterpolatedUnivariateSpline as interp
from fortran_output_analysis.constants_and_parameters import (
    g_inverse_atomic_frequency_to_attoseconds,
    fine_structure,
    g_eV_per_Hartree,
)
from fortran_output_analysis.custom_types import ArrFloat64, ArrComplex128


def l_to_str(l: int) -> str:
    """
    Converts orbital momentum into the corresponding string literal s,p,d ...

    Args:
        l - orbital angular momentum.

    Returns:
        the string literal for the given momentum.
    """
    if l == 0:
        return "s"
    elif l == 1:
        return "p"
    elif l == 2:
        return "d"
    elif l == 3:
        return "f"
    elif l == 4:
        return "g"
    elif l == 5:
        return "h"
    else:
        raise ValueError(
            "l_to_str(): invalid or unimplemented orbital momentum value."
            "Function was given: l =",
            l,
        )


def l_to_int(l: str) -> int:
    """
    Opposite to l_to_str().
    Converts given string literal into the orbital momentum value.

    Args:
        l - string literal for the orbital momentum.

    Returns:
        value of the orbital momentum.
    """

    if l == "s":
        return 0
    elif l == "p":
        return 1
    elif l == "d":
        return 2
    elif l == "f":
        return 3
    elif l == "g":
        return 4
    elif l == "h":
        return 5
    else:
        raise ValueError(
            "l_to_int(): invalid or unimplemented orbital momentum string literal."
            "Function was given: l =",
            l,
        )


def K_to_str(K: int) -> str:
    """
    Converts photon interaction rank into the corresponding string literal S,P,D,F ...

    Args:
        l - orbital angular momentum.

    Returns:
        the string literal for the given momentum.
    """
    if K == 0:
        return "S"
    elif K == 1:
        return "P"
    elif K == 2:
        return "D"
    elif K == 3:
        return "F"
    elif K == 4:
        return "G"
    elif K == 5:
        return "H"
    else:
        raise ValueError(
            "K_to_str(): invalid or unimplemented K value." "Function was given: K =",
            K,
        )


class Hole:

    def __init__(
        self,
        atom_name: str,
        l: int | str,
        n_qn: int,
        binding_energy: float | None = None,
    ) -> None:
        """
        Args:
            atom_name - name of the parent atom.
            l - orbital angular momentum of the hole, given in the form of int or str.
            n_qn - pricnipal quantum number of the hole.
            binding_energy - binding energy for the hole in Hartree. Allows you to specify a
                             predifined alue for the hole's binding energy instead of loading it from
                             the simulation data.
        """
        self.atom_name = atom_name
        if type(l) is str:
            l = l_to_int(l)
        self.l: int = l
        self.n = n_qn
        self.name = construct_hole_name(atom_name, l, n_qn)
        self.binding_energy = binding_energy  # in Hartree

    def load_binding_energy(self, path_to_energies: str) -> None:
        """
        Attempts to load hole's binding energy from the energies.dat file in the output folder.

        Args:
            path_to_energies - path to the energies.dat file.
        """

        try:
            with open(path_to_energies, "r") as f:

                for line in f:
                    line = line.strip()
                    line_split = line.split()

                    # check if we're at the end of the file
                    if int(line_split[0]) == 0 and int(line_split[1]) == 0:
                        raise RuntimeError(
                            "Reached the end of the energies.dat file. The hole's "
                            "binding energy is missing!"
                        )

                    # check for the hole parameters
                    if int(line_split[0]) == self.n and int(line_split[1]) == self.l:
                        bind_en_data = line_split[2]
                        # replace Fortran double-prec notation with Python's e0
                        bind_en_data = bind_en_data.replace("d0", "e0").replace(
                            "D0", "e0"
                        )

                        # locate the binding energy in the output
                        start = bind_en_data.index("(") + 1
                        end = bind_en_data.index(",")
                        bind_en = bind_en_data[start:end]
                        # in Hartree, NOTE the minus sign in front of the float()
                        bind_en = -float(bind_en) / g_eV_per_Hartree
                        self.binding_energy = bind_en
                        return

        except Exception as e:
            print(
                f"{self.name}: Failed to load binding energy from the energies.dat file.! Error: {e}"
            )

        # if we reached this part -> the energy was not loaded -> send a warning
        print(f"Warning: binding energy for the {self.name} hole is not loaded!")


def construct_hole_name(atom_name: str, l: int | str, n_qn: int) -> str:
    """
    Constructs a readable name for the hole with given parameters.

    Args:
        atom_name - name of the parent atom.
        l - orbital angular momentum of the hole.
        n_qn - pricnipal quantum number of the hole.

    Returns:
        name - readable name of the hole.
    """
    name = atom_name + " " + str(n_qn)

    if type(l) is int:
        name += l_to_str(l)
    else:
        name += l

    return name


def assert_out_as(out_as: str) -> None:

    assert out_as in (
        "int",
        "str",
    ), f'The out_as argument must be "int" or "str", not "{out_as}"!'


def load_raw_data(path):
    return np.loadtxt(path)


def extract_data_from_verbose_file(file: str, bp_index: int) -> tuple[
    ArrFloat64,
    ArrComplex128,
    ArrFloat64,
]:
    """
    Extracts matrix elements, along with non-Coulomb phase shifts and final energies, from the
    "verbose" type of output files. Such files contain more descriptive information about the
    calculation process, including, for example, the values of the numerical and analytical parts
    of the two-photon integration.

    Args:
        file - path to the file.
        bp_index - the breakpoint index, starting with 1.

    Returns:
        en_final_arr - array of final energies.
        mat_el_arr - array of matrix elements.
        non_coul_phase_arr - array of non-Coulomb phase shifts.
    """

    en_final_arr = []
    mat_el_arr = []
    non_coul_phase_arr = []

    # the file is read by sections. Each section starts with the " Photon energy" string.
    # Once new section starts, the counter below is used to track the break points.
    bp_counter = -1

    with open(file, "r") as f:
        for line in f:
            # remove the unnecessary spaces/line endings in the beginning and the end of the string
            line = line.strip()

            if line.startswith("Photon energy"):  # new section
                bp_counter = 0
                en_final = np.float64(line.split()[6])
                en_final_arr.append(en_final)

            else:

                if (
                    bp_counter == -1
                ):  # no section has been encountered yet, just continue reading
                    continue

                elif line.startswith("abs") or line.startswith("emi"):
                    bp_counter += 1

                if bp_counter == bp_index:  # if the desired breakpoint is hit
                    data = line.split()

                    mat_el_re = np.float64(data[7])
                    mat_el_im = np.float64(data[8])
                    mat_el = np.complex128(mat_el_re + 1j * mat_el_im)
                    mat_el_arr.append(mat_el)

                    non_coul_phase = np.float64(data[-1])
                    non_coul_phase_arr.append(non_coul_phase)

    en_final_arr = np.array(en_final_arr, dtype=np.float64)
    mat_el_arr = np.array(mat_el_arr, dtype=np.complex128)
    non_coul_phase_arr = np.array(non_coul_phase_arr, dtype=np.float64)

    return en_final_arr, mat_el_arr, non_coul_phase_arr


def phase(x):
    """Returns 1 if the input is even and -1 if it is odd. Mathematically equivalent to (-1)^x"""
    if x % 2 == 0:
        return 1
    else:
        return -1


def mag(x):
    """Returns the absolute value squared of the input"""
    return np.abs(x) ** 2


def cross(x, y):
    """Returns the 'cross term' between x and y: 2Re(x*y^dagger)"""
    return 2 * np.real(x * np.conjugate(y))


def exported_mathematica_tensor_to_python_list(string):
    return json.loads(string.replace("{", "[").replace("}", "]").replace("\n", ""))


def interpolated(x, y, number_of_datapoints):
    """Returns the new x and y values of the 1D-function described by the interpolation
    of the input x and y data points. Interpolates the function values, y, in the domain, x, to
    the given number of datapoints"""
    new_x = np.linspace(x[0], x[-1], number_of_datapoints)
    return new_x, interp(x, y)(new_x)


def l_from_str(l_str):
    l = -1

    if l_str == "s":
        l = 0
    elif l_str == "p":
        l = 1
    elif l_str == "d":
        l = 2
    elif l_str == "f":
        l = 3

    if l == -1:
        raise ValueError(
            "l_from_str(): invalid or unimplemented string for l quantum number."
        )
    else:
        return l


def convert_rate_to_cross_section(rates, omegas, divide=True):
    N = len(omegas)
    cm2 = (0.52917721092**2) * 100.0
    pi = np.pi
    convert_factor = -(1.0 / 3.0) * pi * cm2
    cross_sections = np.zeros(rates.shape)
    omega_factors = np.zeros(len(omegas))
    if divide:
        omega_factors = 1.0 / omegas
    else:
        omega_factors = omegas
    # print(rates[:,2])
    i = 0
    for omega_fac in omega_factors:
        if rates.shape != (N,):
            # print(omega, omega*eV_per_Hartree)
            cross_sections[i, :] = omega_fac * rates[i, :] * convert_factor
        else:
            cross_sections[i] = omega_fac * rates[i] * convert_factor
            # print(cross_sections[i], rates[i])

        i += 1
    return cross_sections


def convert_amplitude_to_cross_section(amplitudes, k, omega, divide_omega=True):
    """
    Computes cross section from the matrix amplitudes and wave numbers.

    Params:
    amplitudes - array with matrix amplitudes
    k - array with wavenumber values
    divide_omega - tells if we divide or multiply by the photon energy (omega) when
    calculating the cross section

    Returns:
    cross_section - array with cross section values
    """

    convert_factor = 2 * np.pi / 3 * fine_structure * 0.52917721092**2 * 100.0
    if divide_omega:
        omega_factor = 1 / omega
    else:
        omega_factor = omega

    cross_section = convert_factor * (amplitudes**2) * k * omega_factor

    return cross_section


def wavenumber(ekin_hart: ArrFloat64) -> ArrFloat64:
    """
    Computes the wavenumber (k-value) in atomic units.

    Args:
        ekin - array of kinetic energies (in Hartree).

    Returns:
        k - vector of wavenumbers.
    """

    ekin_copy = ekin_hart.copy()
    ekin_copy[ekin_copy < 0.0] = (
        0.0  # change all negative elements to 0 to avoid warnings in np.sqrt
    )

    k = np.sqrt(2 * ekin_copy)

    return k


def coulomb_phase(final_l: int | str, ekin_hart: ArrFloat64, Z: int) -> ArrFloat64:
    """
    This is the definition of the phase of the Coulomb function, both the angular momentum part and
    the so-called Coulomb phase. Electron kinetic energy should be given in atomic units.

    Args:
        final_l - orbital angular momentum of the final state, in the int or str format.
        ekin_hart - array of electron kinetic energies (in Hartree).
        Z - charge of the ion.
    """

    if type(final_l) is str:
        final_l = l_to_int(final_l)

    k = wavenumber(ekin_hart)

    nu = Z / k

    b = np.angle(gamma(final_l + 1 + 1j * nu))

    return -b - final_l * np.pi / 2


def delay_to_phase(delay, omega_diff):
    return delay * omega_diff / g_inverse_atomic_frequency_to_attoseconds


def final_energies_for_matching_1sim(energies, steps_per_IR_photon):
    """
    Prepares an array of final energies to match absorption and emission matrices
    in the case of 1 simulation. Simply shifts the energy array by steps_per_IR_photon.

    Params:
    energies - given array of energies
    steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy

    Returns:
    array of final energies for matching
    """

    return energies[steps_per_IR_photon : (len(energies) - steps_per_IR_photon)]


def get_q_res(M, E_res, E_0, width_res):
    """
    Computes Fano shape parameter q for the given resonance.

    Args:
        M - complex matrix element (right eigenvector) corresponding to the resonance.
            The matrix element is typically located in the "diag_matrix_elements_Jtot1.dat" file
            and corresponds to the last two numbers (right eigenvector).
        E_res - resonance position.
        E_0 - ground state energy.
        width_res - width of the resonance.
        NOTE: E_res, E_0, width_res must be in the same units!

    Returns:
        q - the value of the Fano shape parameter.
    """

    M_squar = M * M
    R = np.real(M_squar)
    I = np.imag(M_squar)
    b = ((E_res - E_0) / (width_res / 2) * R / I + 1) / (
        R / I - (E_res - E_0) / (width_res / 2)
    )

    if np.sign(I) > 0:
        q = b - np.sqrt(b**2 + 1)
    else:
        q = b + np.sqrt(b**2 + 1)

    return q


def final_sideband_energies_1sim(
    energies_emi: ArrFloat64,
    energies_abs: ArrFloat64,
    g_omega_IR: float,
    energies_mode: str,
) -> ArrFloat64:
    """
    Prepares an array of final sideband energies in the case of one simulation.

    Args:
        energies_emi - energies of the emission path.
        energies_abs - energies of the absorption path.
        g_omega_IR - energy of the IR photon.
        energies_mode - tells which energies we choose for the final sideband.
                        Possible options:
                        "emi" - emission energies starting from the first available absorption point.
                        "abs" - absorption energies cut at the last available emission point.

    Returns:
        an array of final sideband energies.
    """

    assert energies_mode in (
        "emi",
        "abs",
    ), f"energies_mode for the final sideband energy from one simulation must be 'emi' or 'abs', not '{energies_mode}'!"

    if energies_mode == "emi":
        en_mask = energies_emi >= (
            np.min(energies_emi) + 2 * g_omega_IR
        )  # start from the first absorption point
        return energies_emi[en_mask]
    else:
        en_mask = energies_abs <= (
            np.max(energies_abs) - 2 * g_omega_IR
        )  # cut at the last emission point
        return energies_abs[en_mask]


def final_sideband_energies_2sim(
    energies_emi: ArrFloat64, energies_abs: ArrFloat64, energies_mode: str
) -> ArrFloat64:
    """
    Prepares an array of final sideband energies in the case of 2 simulations.

    Args:
        energies_emi - energies of the emission path.
        energies_abs - energies of the absorption path.
        energies_mode - tells which energies should be taken.
                        Possible options:
                        "emi" - emission energies.
                        "abs" - absorption energies.
                        "both" - combined emission and absorption energies.

    Returns:
        an array of final sideband energies.
    """

    assert energies_mode in (
        "emi",
        "abs",
        "both",
    ), f"energies_mode for the final sideband energy from two simulations must be 'emi', 'abs' or 'both', not '{energies_mode}'!"

    if energies_mode == "emi":
        return energies_emi
    elif energies_mode == "abs":
        return energies_abs
    else:
        energies_concat = np.concatenate((energies_abs, energies_emi))
        energies_final = np.sort(np.unique(energies_concat))
        return energies_final


def match_matrix_elements_1sim(
    energies_final: ArrFloat64,
    energies_emi: ArrFloat64,
    energies_abs: ArrFloat64,
    mat_emi: ArrComplex128,
    mat_abs: ArrComplex128,
) -> tuple[ArrComplex128, ArrComplex128]:
    """
    Matches absoprtion and emission matrix elements to the final sideband energies
    in the case of one simulation.

    Args:
        energies_final - final sideband energies.
        energies_emi - energies of the emission path.
        energies_abs - energies of the absorption path.
        mat_emi - unmatched matrix elements for the emission path.
        mat_abs - unmatched matrix elements for the absorption path.

    Returns:
        mat_emi_matched - matched matrix elements for the emission path.
        mat_abs_matched - matched matrix elements for the absorption path.
    """

    mat_emi_matched = np.interp(energies_final, energies_emi, mat_emi)
    mat_abs_matched = np.interp(energies_final, energies_abs, mat_abs)

    return mat_emi_matched, mat_abs_matched


def match_matrix_elements_2sim(
    energies_final: ArrFloat64,
    energies_emi: ArrFloat64,
    energies_abs: ArrFloat64,
    mat_emi: ArrComplex128,
    mat_abs: ArrComplex128,
    match_mode: str,
) -> tuple[ArrComplex128, ArrComplex128]:
    """
    Matches absoprtion and emission matrix elements to the final sideband energies
    in the case of two simulations.

    Args:
        energies_final - final sideband energies.
        energies_emi - energies of the emission path.
        energies_abs - energies of the absorption path.
        mat_emi - unmatched matrix elements for the emission path.
        mat_abs - unmatched matrix elements for the absorption path.
        match_mode - the mode of the matrix element matching.
                     Possible options:
                     "interp_both" - interpolate both absorption and emission paths for the final
                                     energies.
                     "lin_extrap_emi" - linearly extrapolate the emission path and interpolate the
                                        absorption path for the final energies.
                     "lin_extrap_emi_left" - linearly extrapolate the emission path to the left using
                                             the first two points and interpolate the absosrption
                                             path for the final energies.


    Returns:
        mat_emi_matched - matched matrix elements for the emission path.
        mat_abs_matched - matched matrix elements for the absorption path.
    """

    match_mode_options = (
        "interp_both",
        "lin_extrap_emi",
        "lin_extrap_emi_left",
    )
    assert (
        match_mode in match_mode_options
    ), f"match_mode for the matching of matrices from two simulations must be in {match_mode_options}, not '{match_mode}'!"

    if match_mode == "interp_both":
        mat_emi_matched = np.interp(energies_final, energies_emi, mat_emi)
        mat_abs_matched = np.interp(energies_final, energies_abs, mat_abs)

    elif match_mode == "lin_extrap_emi":
        # linear extrapolation for the emission path
        fit = np.polyfit(energies_emi, mat_emi, 1)  # fitting coefficients
        mat_emi_matched = fit[0] * energies_final + fit[1]
        # interpolation for the absorption path
        mat_abs_matched = np.interp(energies_final, energies_abs, mat_abs)

    elif match_mode == "lin_extrap_emi_left":
        # linearly extrapolate emi to the left, using emi = k * energies + b
        # since extrapolating to the left, k and b are derived using the first two points
        k = (mat_emi[1] - mat_emi[0]) / (energies_emi[1] - energies_emi[0])
        b = mat_emi[0] - k * energies_emi[0]
        mat_emi_matched = k * energies_final + b
        # interpolation for the absorption path
        mat_abs_matched = np.interp(energies_final, energies_abs, mat_abs)

    return mat_emi_matched, mat_abs_matched


def unwrap_phase_with_nans(phase):
    """
    Unwraps phase that contains NaN values by masking out the NaNs.
    """

    # np.unwrap can not handle NaNs, mask them out.
    nanmask = np.logical_not(np.isnan(phase))
    phase[nanmask] = np.unwrap(phase[nanmask])

    return phase


def compute_omega_diff(g_omega_IR_1: float, g_omega_IR_2: float | None = None) -> float:
    """
    Computes energy sepearation between the absorption and emission paths.
    Can compute for 1 or 2 simulations.

    Args:
        g_omega_IR_1 - energy of the IR photon in the first simulation.
        g_omega_IR_2 - energy of the IR photon in the second simulation

    Returns:
        omega_diff - energy difference between the absorption and emission paths.
    """
    if g_omega_IR_2:  # if two simulations are provided
        omega_diff = g_omega_IR_1 + g_omega_IR_2
    else:  # if only one simulation is provided
        omega_diff = 2.0 * g_omega_IR_1

    return omega_diff


def phase_to_delay(phase: ArrFloat64, omega_diff_hart: float) -> ArrFloat64:
    """
    Converts phase into delay.

    Args:
        phase - an array with phase.
        omega_diff_hart - energy separation between the absorption and emission paths in Hartree.

    Returns:
        an array with delay.
    """

    return phase * g_inverse_atomic_frequency_to_attoseconds / omega_diff_hart


def assert_abs_or_emi(abs_or_emi: str) -> None:
    """
    Asserts that the abs_or_emi parameter takes only the allowed values.
    """

    assert abs_or_emi in (
        "abs",
        "emi",
    ), "abs_or_emi parameter can only be 'abs' or 'emi'!"


def extract_data_from_file(file: str, bp_index: int) -> tuple[ArrFloat64, ArrFloat64]:
    """
    Extracts data, corresponding to the specified break point, from the given file.

    Args:
        file - path to the file.
        bp_index - the breakpoint index, starting with 1.

    Returns:
        array of energies and array of values.
    """

    raw = np.loadtxt(file, dtype=np.float64)

    assert bp_index < raw.shape[1], "the break point index is out of range!"

    return raw[:, 0], raw[:, bp_index]


def assert_matrix_shapes(M_emi: np.ndarray, M_abs: np.ndarray) -> None:
    assert (
        M_emi.shape == M_abs.shape
    ), "The shapes of the absorption and emission matrices are not the same!"
