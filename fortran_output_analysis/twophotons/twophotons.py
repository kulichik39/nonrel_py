import numpy as np

from fortran_output_analysis.common_utility import (
    l_to_str,
    l_to_int,
    assert_abs_or_emi,
    K_to_str,
)
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree
from fortran_output_analysis.types import (
    ArrFloat64,
    ArrFloat64_2D,
    ArrFloat64_3D,
    ArrComplex128,
    ArrComplex128_3D,
)

from fortran_output_analysis.common_utility import (
    Hole,
    assert_out_as,
    construct_hole_name,
)

from fortran_output_analysis.twophotons.utilities import (
    extract_data_from_verbose_file,
)

"""
NOTE: The names of some classes below are not perfect and could be changed for 
better code readabiltiy. However, it was decided to keep them untouched to maintain similarity with
the older relativistic code. I tried to add a more informative description to each class to make 
things a little clearer.
"""


def final_ls(hole_l: int, only_reachable: bool = True) -> list[int]:
    """
    If only_reachable is True, returns final angular momenta that, assuming dipole approximation,
    can be reached with two photons from an initial state specified by hole_l.
    If only_reachable is False, always returns a list of three elements in a specific order
    ([hole_l - 2, hole_l, hole_l + 2]), where some of the values may correspond to non-existing
    channels (e.g. l=-2 for the s hole). This option is needed to set the data dimensionality.

    Args:
        hole_l - angular momentum of the hole.
        only_reachable - tells if only dipole allowed final states should be returned.

    Returns:
        l_final - list of final angular momenta.
    """

    l_final = [hole_l - 2, hole_l, hole_l + 2]

    if only_reachable:
        l_final = [l for l in l_final if l >= 0]

    return l_final


def get_Ks(
    hole_l: int,
    final_l: int,
    only_reachable: bool = True,
    light_polar: str = "lin",
) -> list[int]:
    """
    If only_reachable is True, returns the ranks of the two-photon interaction that contribute to
    the coupling of the initial (specified by hole_l) and final (specified by final_l) states,
    assuming the provided light polarization. NOTE: for now, only the linear polarization is
    implemented.

    If only_reachable is False, always returns a list of three values [0, 1, 2]. This option is
    needed to set the data dimensionality.

    Args:
        hole_l - angular momentum of the hole.
        final_l - angular momentum of the final state.
        only_reachable - tells if only allowed K values should be returned.
        light_polar - specifies light polarization.
                      Possible options:
                      "lin" - linear.

    Returns:
        K_values - the list with the ranks of photon interaction.
    """

    light_polar_options = ("lin",)
    assert (
        light_polar in light_polar_options
    ), f"The light polarization should be in {light_polar_options}!"

    K_values = [0, 1, 2]

    if only_reachable:
        K_values_copy = K_values.copy()
        K_values = []  # new list to store reachable Ks
        for K in K_values_copy:
            if (K >= np.abs(final_l - hole_l)) and (K <= final_l + hole_l):
                if light_polar == "lin" and K == 1:
                    continue  # K = 1 doesn't contribute in the case of linear polarization
                else:
                    K_values.append(K)

    return K_values


class IonisationPath:
    """
    Stores information about an ionisation path (channel).
    NOTE: here and in the following, the terms "ionisation path" and "ionisation channel" are used
    interchangebly. The name of this class was kept as IonisationPath to maintain similariy with
    the relativistic code.
    """

    def __init__(self, l: int, K: int, row_idx: int) -> None:
        """
        Args:
            l - orbital angular momentum of the final state.
            K - the rank of the two-photon interaction.
            row_idx - index of the row in the data corresponding to this ionisation path.
        """
        self.l: int = l
        self.K = K
        self.name = "-> " + l_to_str(self.l) + f" with K={K}"

        self.row_index = row_idx
        self.K_index = K


class Channels:
    """
    Represents hole-specific simulation data. Allows to store/manipulate ionisation channels for
    a hole as well as the relevant output data (amplitude, phase, etc).
    """

    def __init__(
        self,
        hole: Hole,
        data_files: dict[tuple[int | str, int], str],
        bp_index: int,
    ) -> None:
        """
        Args:
            hole - an object of the Hole class.
            data_files - a dictionary mapping ionisation channels to the corresponding data files.
                         The structure of the dict is as follows:
                                    {
                                        (final_l_1, K_1): file_1
                                        ,
                                        (final_l_2, K_2): file_2
                                        ,
                                        ...
                                    }
            bp_index - the breakpoint index, starting with 1.
        """

        self.hole: Hole = hole
        self._ionisation_channels: dict[tuple[int, int], IonisationPath] = {}
        self._omega_2ph_data: ArrFloat64 | None = (
            None  # combined energy of 2 photons in Hartree
        )
        self._mat_el_data: ArrComplex128_3D | None = None  # in a.u.
        self._phase_data: ArrFloat64_3D | None = None  # in a.u.
        self._set_ionisation_data(data_files, bp_index)

    def _set_ionisation_data(
        self, data_files: dict[tuple[int | str, int], str], bp_index: int
    ) -> None:
        """
        Sets up the reachable ionisation channels and stores the relevant data in the attributes.

        NOTE: The first dimension of the amplitude and phase arrays is fixed at the maximum
        number of final l values returned by final_ls(), regardless of the actual number of
        dipole-allowed (reachable) final states. The second dimension is fixed at the maximum
        number of K values returned by get_Ks().
        This fixed structure is required to:
        1. Calculate the beta parameters, which rely on a specific ordering of the final states,
        K values, and their corresponding coefficients.
        2. Maintain consistency with the older relativistic code.

        Args:
            data_files - a dictionary mapping ionisation channels to the corresponding data files.
            bp_index - the breakpoint index, starting with 1.
        """
        hole_l = self.hole.l

        all_final_l = final_ls(hole_l, only_reachable=False)
        N_l = len(all_final_l)  # the max. number of final states

        # the final l value (0) and light polarization doesn't matter here since we
        # want to get the maximum number of K values
        all_K = get_Ks(hole_l, 0, only_reachable=False, light_polar="lin")
        N_K = len(all_K)

        # load two photon energies from the first file that came across
        first_file = next(iter(data_files.values()))
        omega_2ph_eV, _, _ = extract_data_from_verbose_file(first_file, 1)
        omega_2ph_hart = omega_2ph_eV / g_eV_per_Hartree
        self._omega_2ph_data = omega_2ph_hart
        N_omega = len(omega_2ph_hart)

        # initialize all the other data arrays
        self._mat_el_data = np.zeros((N_l, N_K, N_omega), dtype=np.complex128)
        self._phase_data = np.zeros((N_l, N_K, N_omega), dtype=np.float64)

        # get the type of the angular momenta in the data_files keys (int or str), and transform
        # them into the int format if needed.
        # since in the 2ph case the keys are tuples, we need to access the first element which
        # corresponds to the final l
        type_l_keys = type(next(iter(data_files.keys()))[0])
        if type_l_keys is str:
            data_files = {
                (l_to_int(key[0]), key[1]): value for key, value in data_files.items()
            }

        reachable_final_l = final_ls(
            hole_l, only_reachable=True
        )  # list of reachable final states

        for idx in range(N_l):
            final_l = all_final_l[idx]

            # save only the reachable final states
            if final_l in reachable_final_l:
                reachable_K = get_Ks(
                    hole_l, final_l, only_reachable=True, light_polar="lin"
                )
                for K in reachable_K:
                    try:
                        file = data_files[(final_l, K)]
                    except KeyError:
                        raise KeyError(
                            f"The {l_to_str(final_l)} final state with K={K} is reachable, but was not found in the data files!"
                        ) from None  # "from None" means that the previous exception log will be hidden

                    # store the channel's data
                    self._ionisation_channels[(final_l, K)] = IonisationPath(
                        final_l, K, idx
                    )
                    _, mat_el, phase = extract_data_from_verbose_file(file, bp_index)
                    self._mat_el_data[idx, K] = mat_el
                    self._phase_data[idx, K] = phase

    def _assert_ionisation_channel(self, final_l: int, K: int) -> None:
        """
        Checks if the given ionisation channel, determined by the final angular momentum l and
        the rank of the two-photon interaction K, is available.

        Args:
            final_l - orbital momentum of the final state.
            K - the rank of the photon interaction.
        """

        assert (
            final_l,
            K,
        ) in self._ionisation_channels, f"The {l_to_str(final_l)} state with K={K} is not reachable from the {self.hole.name} hole after 2 photons!"

    def get_ionisation_channel(self, final_l: int | str, K: int) -> IonisationPath:
        """
        Args:
            final_l - orbital momentum of the final state, given in the form of int or str.
            K - the rank of the photon interaction.

        Returns:
            an IonisationPath instance for the specified ionisation channel.
        """

        if type(final_l) is str:
            final_l = l_to_int(final_l)

        self._assert_ionisation_channel(final_l, K)

        return self._ionisation_channels[(final_l, K)]

    def get_all_ionisation_channels(self) -> list[IonisationPath]:
        """
        Returns:
            the list of all the ionisation channels.
        """

        return list(self._ionisation_channels.values())

    def get_l_K_to_indices(
        self, out_as: str = "str"
    ) -> dict[tuple[int | str, int], tuple[int, int]]:
        """
        Args:
            out_as - whether to output orbital momenta in the form of int or str.

        Returns:
            a dictionary mapping reachable orbital momenta l and ranks K with the indices in the
            data arrays. The structure of the dict is as follows:
                                            {
                                               (final_l_1, K_1): (row_idx_1, K_idx_1),
                                               (final_l_2, K_2): (row_idx_2, K_idx_2),
                                               ...
                                            }
        """

        assert_out_as(out_as)

        l_K_to_indices = {}

        final_l_reachable = final_ls(self.hole.l, only_reachable=True)

        for final_l in final_l_reachable:
            K_reachable = get_Ks(
                self.hole.l, final_l, only_reachable=True, light_polar="lin"
            )
            for K in K_reachable:
                ion_channel = self.get_ionisation_channel(final_l, K)

                if out_as == "str":  # turn the angular momentum into the out_as format
                    l_key = l_to_str(final_l)
                else:
                    l_key = final_l

                l_K_to_indices[(l_key, K)] = (
                    ion_channel.row_index,
                    ion_channel.K_index,
                )

        return l_K_to_indices

    def get_omega_2ph(self) -> ArrFloat64:
        """
        Returns:
            an array of final energies (after 2 ph).
        """

        return self._omega_2ph_data

    def get_mat_el_one_channel(self, final_l: int | str, K: int) -> ArrComplex128:
        """
        Args:
            final_l - orbital momentum of the final state, given in the form of int or str.
            K - the rank of the photon interaction.

        Retutns:
            matrix elements for the given ionisation channel.
        """

        channel = self.get_ionisation_channel(final_l, K)
        row_idx = channel.row_index
        K_idx = channel.K_index

        return self._mat_el_data[row_idx, K_idx]

    def get_mat_el(self) -> ArrComplex128_3D:
        """
        Retutns:
            matrix elements for all ionisation channels.
        """

        return self._mat_el_data

    def get_phase_one_channel(self, final_l: int | str, K: int) -> ArrFloat64:
        """
        Args:
            final_l - orbital momentum of the final state, given in the form of int or str.
            K - the rank of the photon interaction.

        Retutns:
            non-coulomb phase for the given ionisation channel.
        """

        channel = self.get_ionisation_channel(final_l, K)
        row_idx = channel.row_index
        K_idx = channel.K_index

        return self._phase_data[row_idx, K_idx]

    def get_phase(self) -> ArrFloat64_3D:
        """
        Retutns:
            non-coulomb phase for all ionisation channels.
        """

        return self._phase_data


class TwoPhotons:
    """
    Provides a central interface for storing, accessing, and manipulating
    two photon simulation data.

    The class supports multiple holes and two ionisation pathways: absorption
    and emission. For each hole, data may be loaded for either pathway or for
    both. Each hole-pathway combination is represented by an instance of the
    Channels class, which manages the corresponding data.
    """

    def __init__(self, atom_name: str, g_omega_IR_hart: float) -> None:
        """
        Args:
            atom_name - name of the atom.
            g_omega_IR_hart - energy of the IR photon in Hartree used in the simulation.
        """

        self.atom_name = atom_name
        self.g_omega_IR_hart = g_omega_IR_hart  # energy of the IR photon in Hartree
        self._channels_abs = {}
        self._channels_emi = {}

    def is_hole_loaded(self, abs_or_emi: str, n_qn: int, hole_l: int | str) -> bool:
        """
        Checks if the hole is loaded.

        Args:
            abs_or_emi - which path to check for: absroption or emission.
            n_qn - principal quantum number of the hole.
            hole_l - orbital angular momentum of the hole, in the int or str format.

        Returns:
            True if loaded, False otherwise.
        """

        assert_abs_or_emi(abs_or_emi)

        if type(hole_l) is str:  # convert into int, if given in the str format
            hole_l = l_to_int(hole_l)

        if abs_or_emi == "abs":
            return (n_qn, hole_l) in self._channels_abs

        elif abs_or_emi == "emi":
            return (n_qn, hole_l) in self._channels_emi

    def load_hole(
        self,
        abs_emi_or_both: str,
        n_qn: int,
        hole_l: int | str,
        with_shake_ups: bool = False,
        path_to_data: str | None = None,
        data_files_for_path: dict[str, dict[tuple[int | str, int], str]] | None = None,
        bp_index: int = 3,
        should_reload: bool = False,
        binding_energy: float | None = None,
        path_to_energies: str | None = None,
    ) -> None:
        """
        Initializes a hole, corresponding ionisation channels, and loads simulation data for it.
        The data is stored in an instance of the Channels class.

        Args:
            abs_emi_or_both - tells which pathway we load the data for: absorption, emission or both.
            n_qn - principal quantum number of the hole.
            hole_l - orbital momentum of the hole, in the int or str format.
            path_to_data - path to the output folder with the simulation results.
            with_shake_ups - tells if the shake ups were included in the calculations. Needed because
                             the names of the output files depend on that.

            data_files_for_paths - a dictionary mapping an ionisation pathway ("abs" or "emi")
                                   to another dictionary that maps ionisation channels to their
                                   data files (i.e. data_files for the Channels class constructor).
                                   If not specified, constructed using path_to_data and assuming
                                   standard naming of the output files.
                                   The structure of the dict is as follows:
                                                    {
                                                     "abs": {
                                                             (final_l_1, K_1): file_1
                                                             ,
                                                             (final_l_2, K_2): file_2
                                                             ,
                                                             ...
                                                            }
                                                     ,
                                                     "emi": {
                                                             (final_l_1, K_1): file_1
                                                             ,
                                                             (final_l_2, K_2): file_2
                                                             ,
                                                             ...
                                                            }
                                                    }

            bp_index - the breakpoint index, starting with 1.
            should_reload - in case the data was previously loaded, tells if it should be reloaded.
            binding_energy - binding energy for the hole. Allows specifying the predifined value for
            the hole's binding energy instead of loading it from the simulation data.
            path_to_energies - path to the energies.dat file with the binding energies.
        """

        assert abs_emi_or_both in (
            "abs",
            "emi",
            "both",
        ), "abs_emi_or_both parameter can only be 'abs', 'emi' or 'both'!"

        if abs_emi_or_both == "both":
            is_loaded = self.is_hole_loaded(
                "abs", n_qn, hole_l
            ) and self.is_hole_loaded("emi", n_qn, hole_l)

        else:
            is_loaded = self.is_hole_loaded(abs_emi_or_both, n_qn, hole_l)

        if not is_loaded or should_reload:

            if type(hole_l) is str:  # convert into int, if given in the str format
                hole_l = l_to_int(hole_l)

            hole = Hole(
                self.atom_name, hole_l, n_qn, binding_energy=binding_energy
            )  # initialize hole object

            if is_loaded:
                print(
                    f"Reload the two photon data for the {hole.name} hole in {self.atom_name}!"
                )

            # If the paths to the data files were not specified, construct them assuming the
            # standard naming.
            if not data_files_for_path:
                data_files_for_path = {}

                if abs_emi_or_both == "abs" or abs_emi_or_both == "both":
                    data_files_for_path["abs"] = {}

                if abs_emi_or_both == "emi" or abs_emi_or_both == "both":
                    data_files_for_path["emi"] = {}

                final_l_reachable = final_ls(
                    hole_l, only_reachable=True
                )  # list of reachable final states

                for final_l in final_l_reachable:

                    reachable_K = get_Ks(
                        hole_l, final_l, only_reachable=True, light_polar="lin"
                    )

                    for K in reachable_K:
                        hole_l_str = l_to_str(hole_l)
                        final_l_str = l_to_str(final_l)
                        K_str = K_to_str(K)

                        # specify data files for the absorption path
                        if abs_emi_or_both == "abs" or abs_emi_or_both == "both":
                            if with_shake_ups:
                                raise RuntimeError(
                                    "The data_files set up when the shake ups are included is not yet implemented in the two photon case!"
                                )
                            else:
                                data_files_for_path["abs"][(final_l, K)] = (
                                    path_to_data
                                    + f"abs_{n_qn}{hole_l_str}{final_l_str}{K_str}_0.dat"
                                )

                        # specify data files for the emission path
                        if abs_emi_or_both == "emi" or abs_emi_or_both == "both":
                            if with_shake_ups:
                                raise RuntimeError(
                                    "The data_files set up when the shake ups are included is not yet implemented in the two photon case!"
                                )
                            else:
                                data_files_for_path["emi"][(final_l, K)] = (
                                    path_to_data
                                    + f"emi_{n_qn}{hole_l_str}{final_l_str}{K_str}_0.dat"
                                )
            if (
                not binding_energy
            ):  # if the value for binding energy hasn't been provided - load it from data

                if not path_to_energies:
                    path_to_energies = path_to_data + "energies.dat"

                hole.load_binding_energy(path_to_energies)

            # store the hole's information and load data for ionisation paths
            if abs_emi_or_both == "abs" or abs_emi_or_both == "both":
                data_files_abs = data_files_for_path["abs"]
                self._channels_abs[(n_qn, hole_l)] = Channels(
                    hole, data_files_abs, bp_index
                )

            if abs_emi_or_both == "emi" or abs_emi_or_both == "both":
                data_files_emi = data_files_for_path["emi"]
                self._channels_emi[(n_qn, hole_l)] = Channels(
                    hole, data_files_emi, bp_index
                )

    def assert_hole_load(self, n_qn: int, hole_l: int | str):
        """
        Asserts that the hole was loaded.

        Args:
            n_qn - principal quantum number of the hole.
            hole_l - orbital angular momentum of the hole, in the int or str format.
        """

        assert self.is_hole_loaded(
            n_qn, hole_l
        ), f"The {construct_hole_name(self.atom_name, hole_l, n_qn)} hole is not loaded!"

    def get_channels_for_hole(self, n_qn: int, hole_l: int | str) -> Channels:
        """
        Returns an instance of the Channels class for the given hole.
        The instance contains all the relevant hole data.

        Args:
            n_qn - principal quantum number of the hole.
            hole_l - orbital angular momentum of the hole, in the int or str format.

        Returns:
            an instance of the Channels class for the given hole
        """

        self.assert_hole_load(n_qn, hole_l)

        if type(hole_l) is str:  # convert into int, if given in the str format
            hole_l = l_to_int(hole_l)

        return self._channels[(n_qn, hole_l)]

    def get_all_channels(self) -> list[Channels]:
        """
        Returns:
            Channels instances for all loaded holes.
        """

        return list(self._channels.values())

    def get_loaded_hole(self, n_qn: int, hole_l: int | str) -> Hole:
        """
        Args:
            n_qn - principal quantum number of the hole.
            hole_l - orbital angular momentum of the hole, in the int or str format.

        Returns:
            a instance of the Hole class corresponding to the loaded hole.
        """

        channels = self.get_channels_for_hole(n_qn, hole_l)

        return channels.hole

    def get_all_loaded_holes(self) -> list[Hole]:
        """
        Returns:
            a list of all loaded hole objects.
        """

        channels_list = self.get_all_channels()

        return [channels.hole for channels in channels_list]

    def get_channel_labels_for_hole(self, n_qn: int, hole_l: int | str) -> list[str]:
        """
        Constructs labels for all ionisation channels of the given hole.

        Args:
            n_qn - principal quantum number of the hole.
            hole_l - orbital angular momentum of the hole, in the int or str format.

        Returns:
            channel_labels - list with labels of all ionisation channels.
        """

        channels = self.get_channels_for_hole(n_qn, hole_l)
        hole_name = channels.hole.name
        ionisation_channels = channels.get_all_ionisation_channels()

        return [
            hole_name + " " + ion_channel.name for ion_channel in ionisation_channels
        ]
