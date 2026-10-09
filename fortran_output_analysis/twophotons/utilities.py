import numpy as np
from fortran_output_analysis.types import ArrFloat64, ArrComplex128


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
