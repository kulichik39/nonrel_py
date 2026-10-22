import numpy as np

from fortran_output_analysis.common_utility import (
    l_to_str,
    l_to_int,
    Hole,
    extract_data_from_file,
    assert_out_as,
    construct_hole_name,
)

from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree
from fortran_output_analysis.custom_types import ArrFloat64, ArrFloat64_2D

"""
NOTE: The names of some classes below are not perfect and could be changed for 
better code readabiltiy. However, it was decided to keep them untouched to maintain similarity with
the older relativistic code. I tried to add a more informative description to each class to make 
things a little clearer.
"""


def final_ls(hole_l: int, only_reachable: bool = True) -> list[int]:
    """
    If only_reachable is True, returns final angular momenta that, assuming dipole approximation,
    can be reached with one photon from an initial state specified by hole_l.
    If only_reachable is False, always returns a list of two elements in a specific order
    ([hole_l - 1, hole_l + 1]), where one of the values may correspond to a non-existing channel
    (e.g. l=-1 for the s hole).


    Args:
        hole_l - angular momentum of the hole.
        only_reachable - tells if only dipole allowed final states should be returned.

    Returns:
        l_final - list of final angular momenta.
    """

    l_final = [hole_l - 1, hole_l + 1]

    if only_reachable:
        l_final = [l for l in l_final if l >= 0]

    return l_final


class IonisationPath:
    """
    Stores information about an ionisation path (channel).
    NOTE: here and in the following, the terms "ionisation path" and "ionisation channel" are used
    interchangebly. The name of this class was kept as IonisationPath to maintain similariy with
    the relativistic code.
    """

    def __init__(self, l: int, row_idx: int) -> None:
        """
        Args:
            l - orbital angular momentum of the final state.
            row_idx - index of the row in the data corresponding to this ionisation path.
        """
        self.l: int = l
        self.name = "-> " + l_to_str(self.l)
        self.row_index = row_idx


class Channels:
    """
    Represents hole-specific simulation data. Allows to store/manipulate ionisation channels for
    a hole as well as the relevant output data (cross section, amplitude, phase, etc).
    """

    def __init__(
        self,
        hole: Hole,
        data_files: dict[int | str, dict[str, str]],
        bp_index: int,
    ) -> None:
        """
        Args:
            hole - an object of the Hole class.
            data_files - a dictionary mapping ionisation channels to the corresponding data files.
            bp_index - the breakpoint index, starting with 1.
        """

        self.hole: Hole = hole
        self._ionisation_channels: dict[int, IonisationPath] = {}
        self._omega_data: ArrFloat64 | None = None  # in Hartree
        self._cs_data: ArrFloat64_2D | None = None  # in Mb
        self._amp_data: ArrFloat64_2D | None = None  # in a.u.
        self._phase_data: ArrFloat64_2D | None = None  # in a.u.
        self._set_ionisation_data(data_files, bp_index)

    def _set_ionisation_data(
        self, data_files: dict[int | str, dict[str, str]], bp_index: int
    ) -> None:
        """
        Sets up reachable ionisation channels and saves relevant data in the attributes.

        NOTE: The first dimension of the cs, amplitude, and phase arrays is fixed at the maximum
        number of final l values returned by final_ls(), regardless of the actual number of
        dipole-allowed (reachable) final states. This fixed structure is required to:
        1. Calculate the beta parameters, which rely on a specific ordering of
        the final states and their corresponding coefficients.
        2. Maintain consistency with the older relativistic code.

        Args:
            data_files - a dictionary mapping ionisation channels to the corresponding data files.
            bp_index - the breakpoint index, starting with 1.
        """
        hole_l = self.hole.l

        all_final_l = final_ls(hole_l, only_reachable=False)
        N_l = len(all_final_l)  # the max. number final states

        # load photon energies from the first cross section file that came across
        first_cs_file = next(iter(data_files.values()))["cs"]
        omega_eV, _ = extract_data_from_file(first_cs_file, 1)
        omega_hart = omega_eV / g_eV_per_Hartree
        self._omega_data = omega_hart
        N_omega = len(omega_hart)

        # initialize all the other data arrays
        self._cs_data = np.zeros((N_l, N_omega), dtype=np.float64)
        self._amp_data = np.zeros((N_l, N_omega), dtype=np.float64)
        self._phase_data = np.zeros((N_l, N_omega), dtype=np.float64)

        # get the type of the angular momenta in the data_files keys (int or str), and transform
        # them into the int format if needed
        type_l_keys = type(next(iter(data_files.keys())))
        if type_l_keys is str:
            data_files = {l_to_int(key): value for key, value in data_files.items()}

        reachable_final_l = final_ls(
            hole_l, only_reachable=True
        )  # list of reachable final states

        for idx in range(N_l):
            final_l = all_final_l[idx]

            # save only the reachable final states
            if final_l in reachable_final_l:
                try:
                    files = data_files[final_l]
                except KeyError:
                    raise KeyError(
                        f"The {l_to_str(final_l)} final state is reachable, but was not found in the data files!"
                    ) from None  # "from None" means that the previous exception log will be hidden

                # store the channel's data
                self._ionisation_channels[final_l] = IonisationPath(final_l, idx)
                # cross section
                cs_file = files["cs"]
                _, cs = extract_data_from_file(cs_file, bp_index)
                self._cs_data[idx] = cs
                # amplitude
                amp_file = files["amp"]
                _, amp = extract_data_from_file(amp_file, bp_index)
                self._amp_data[idx] = amp
                # phase
                phase_file = files["phase"]
                _, phase = extract_data_from_file(phase_file, bp_index)
                self._phase_data[idx] = phase

    def _assert_ionisation_channel(self, final_l: int) -> None:
        """
        Checks if the given ionisation channel, determined by the final angular momentum l,
        is available.

        Args:
            final_l - orbital momentum of the final state.
        """

        assert (
            final_l in self._ionisation_channels
        ), f"The {l_to_str(final_l)} state is not reachable from the {self.hole.name} hole after 1 photon!"

    def get_ionisation_channel(self, final_l: int | str) -> IonisationPath:
        """
        Args:
            final_l - orbital momentum of the final state, given in the form of int or str.

        Returns:
            an IonisationPath instance for the specified ionisation channel.
        """

        if type(final_l) is str:
            final_l = l_to_int(final_l)

        self._assert_ionisation_channel(final_l)

        return self._ionisation_channels[final_l]

    def get_all_ionisation_channels(self) -> list[IonisationPath]:
        """
        Returns:
            the list of all the ionisation channels.
        """

        return list(self._ionisation_channels.values())

    def get_l_to_index(self, out_as: str = "str") -> dict[int | str, int]:
        """
        Maps final orbital momenta with the corresponding row indices in the data arrays.

        Args:
            out_as - whether to output final orbital momenta in the form of int or str.

        Returns:
            a dictionary mapping reachable final momenta with the row indices in the data.
        """

        assert_out_as(out_as)

        l_to_index = {}

        # get the list of all ionisation channels (paths)
        ion_channel_list = self.get_all_ionisation_channels()

        for ion_channel in ion_channel_list:
            l_key = ion_channel.l  # int
            if out_as == "str":  # turn the dictionary key into the out_as format
                l_key = l_to_str(l_key)

            l_to_index[l_key] = ion_channel.row_index

        return l_to_index

    def get_omega(self) -> ArrFloat64:
        """
        Returns:
            XUV photon energies used in the simulation.
        """

        return self._omega_data

    def get_cs_one_channel(self, final_l: int | str) -> ArrFloat64:
        """
        Args:
            final_l - orbital momentum of the final state, given in the form of int or str.

        Retutns:
            cross section for the given ionisation channel.
        """

        channel = self.get_ionisation_channel(final_l)
        row_idx = channel.row_index

        return self._cs_data[row_idx]

    def get_cs(self) -> ArrFloat64_2D:
        """
        Retutns:
            cross section for all ionisation channels.
        """

        return self._cs_data

    def get_amp_one_channel(self, final_l: int | str) -> ArrFloat64:
        """
        Args:
            final_l - orbital momentum of the final state, given in the form of int or str.

        Retutns:
            amplitude for the given ionisation channel.
        """

        channel = self.get_ionisation_channel(final_l)
        row_idx = channel.row_index

        return self._amp_data[row_idx]

    def get_amp(self) -> ArrFloat64_2D:
        """
        Retutns:
            amplitude for all ionisation channels.
        """

        return self._amp_data

    def get_phase_one_channel(self, final_l: int | str) -> ArrFloat64:
        """
        Args:
            final_l - orbital momentum of the final state, given in the form of int or str.

        Retutns:
            phase for the given ionisation channel.
        """

        channel = self.get_ionisation_channel(final_l)
        row_idx = channel.row_index

        return self._phase_data[row_idx]

    def get_phase(self) -> ArrFloat64_2D:
        """
        Retutns:
            phase for all ionisation channels.
        """

        return self._phase_data


class OnePhoton:
    """
    Provides a central interface for storing, accessing, and manipulating
    one photon simulation data, including both general (like eigenstates from
    diagonalisation) and hole-specific (like amplitude, phase etc.) results.

    The class supports multiple holes. Each hole is attributed to an instance of
    the Channels class, which manages the corresponding data.
    """

    def __init__(self, atom_name: str) -> None:
        """
        Args:
            atom_name - name of the atom.
        """

        # attributes for diag data
        self._diag_eigenvalues = None  # in Hartree
        self._diag_loaded = False  # tells whether diagonal data was loaded

        # attributes for holes' data
        self.atom_name = atom_name
        self._channels = {}

    def load_diag_data(
        self,
        path_to_data: str | None = None,
        path_to_diag_eigenvalues: str | None = None,
        should_reload: bool = False,
    ) -> None:
        """
        Loads the results of the RPAE matrix diagonalisation. NOTE: for now, the eigenvalues only,
        without the matrix elements!

        Args:
            path_to_data - path to the output folder with the simulation results.
            path_to_diag_eigenvalues - path to the file with the eigenvalues.
            should_reload - in case the data was previously loaded, tells if it should be reloaded.
        """

        if not self._diag_loaded or should_reload:
            if self._diag_loaded:
                print(f"Reload diagonal data in {self.atom_name}!")

            # if the paths to diag data are not specified, we assume the standard names in the
            # output data folder
            if not path_to_diag_eigenvalues:
                path_to_diag_eigenvalues = path_to_data + "ErEi_1P.dat"
            self.__load_diag_eigenvalues(path_to_diag_eigenvalues)

            self._diag_loaded = True

    def __load_diag_eigenvalues(self, path_to_diag_eigenvalues: str) -> None:
        """
        Loads eigenvalues from the RPAE matrix diagonalization.

        Args:
            path_to_diag_eigenvalues - path to the file with the eigenvalues.
        """

        eigenvals_raw = np.loadtxt(path_to_diag_eigenvalues)
        eigenvals_raw = eigenvals_raw[:, :2]
        eigvals_re = eigenvals_raw[:, 0]  # real part
        eigvals_im = eigenvals_raw[:, 1]  # imaginary part
        self._diag_eigenvalues = eigvals_re + 1j * eigvals_im

    def _assert_diag_data_load(self) -> None:
        """
        Assertion that the diagonalisation data was loaded.
        """
        assert (
            self._diag_loaded
        ), f"Diagonalisation data is not loaded for {self.atom_name}!"

    def get_diag_eigenvalues(self) -> ArrFloat64:
        """
        Retruns:
            eigenvalues from the RPAE matrix diagonalisation.
        """

        self._assert_diag_data_load()

        return self._diag_eigenvalues

    def is_hole_loaded(self, n_qn: int, hole_l: int | str) -> bool:
        """
        Checks if the hole is loaded.

        Args:
            n_qn - principal quantum number of the hole.
            hole_l - orbital angular momentum of the hole, in the int or str format.

        Returns:
            True if loaded, False otherwise.
        """

        if type(hole_l) is str:  # convert into int, if given in the str format
            hole_l = l_to_int(hole_l)

        return (n_qn, hole_l) in self._channels

    def load_hole(
        self,
        n_qn: int,
        hole_l: int | str,
        path_to_data: str | None = None,
        data_files: dict[int | str, dict[str, str]] | None = None,
        bp_index: int = 3,
        should_reload: bool = False,
        binding_energy: float | None = None,
        path_to_energies: str | None = None,
    ) -> None:
        """
        Initializes a hole, corresponding ionisation channels, and loads simulation data for it.
        The data is stored in an instance of the Channels class.

        Args:
            n_qn - principal quantum number of the hole.
            hole_l - orbital momentum of the hole.
            path_to_data - path to the output folder with the simulation results.

            data_files - a dictionary mapping ionisation channels to the corresponding data files.
                         If not specified, constructed using path_to_data and assuming standard
                         naming of the output files. The structure of the dict is as follows:
                                                    {
                                                        final_l_1: {
                                                                    "cs": cs_file_1,
                                                                    "amp": amp_file_1,
                                                                    "phase": phase_file_1
                                                                    }
                                                        ,
                                                        final_l_2: {
                                                                    "cs": cs_file_2,
                                                                    "amp": amp_file_2,
                                                                    "phase": phase_file_2
                                                                    }
                                                        ,
                                                        ...
                                                    }

            bp_index - the breakpoint index, starting with 1.
            should_reload - in case the data was previously loaded, tells if it should be reloaded.
            binding_energy - binding energy for the hole in Hartree. Allows specifying the
                             predifined value for the hole's binding energy instead of loading it
                             from the simulation data.
            path_to_energies - path to the energies.dat file with the binding energies.
        """
        is_loaded = self.is_hole_loaded(n_qn, hole_l)

        if not is_loaded or should_reload:

            if type(hole_l) is str:  # convert into int, if given in the str format
                hole_l = l_to_int(hole_l)

            hole = Hole(
                self.atom_name, hole_l, n_qn, binding_energy=binding_energy
            )  # initialize hole object

            if is_loaded:
                print(
                    f"Reload the one photon data for the {hole.name} hole in {self.atom_name}!"
                )

            # If the paths to the data files were not specified, construct them assuming the
            # standard naming.
            if not data_files:
                data_files = {}
                reachable_final_l = final_ls(
                    hole_l, only_reachable=True
                )  # list of reachable final states
                for final_l in reachable_final_l:
                    data_files[final_l] = {}
                    hole_l_str = l_to_str(hole_l)
                    final_l_str = l_to_str(final_l)
                    data_files[final_l]["cs"] = (
                        path_to_data + f"cross_{n_qn}{hole_l_str}_to_{final_l_str}.dat"
                    )
                    if final_l < hole_l:
                        data_files[final_l]["amp"] = (
                            path_to_data + "amplitude_ldown.dat"
                        )
                        data_files[final_l]["phase"] = path_to_data + "phase_ldown.dat"
                    else:
                        data_files[final_l]["amp"] = path_to_data + "amplitude_lup.dat"
                        data_files[final_l]["phase"] = path_to_data + "phase_lup.dat"

            if (
                not binding_energy
            ):  # if the value for binding energy hasn't been provided - load it from data

                if not path_to_energies:
                    path_to_energies = path_to_data + "energies.dat"

                hole.load_binding_energy(path_to_energies)

            # store the hole's information and load data for ionisation channels
            self._channels[(n_qn, hole_l)] = Channels(hole, data_files, bp_index)

    def assert_hole_load(self, n_qn: int, hole_l: int | str) -> None:
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
        The instance contains all the relevant simulation data.

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
