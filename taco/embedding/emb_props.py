"""Properties evaluated post-embedding calculations."""

import csv
from taco.log import log
import numpy as np


class EmbProp():
    """PySCF wrapper for embedding calculations.

    Attributes
    ----------
    emb_pot : EmbPot object
        Embedding potential object.

    Methods
    -------
    __init__(self, emb_pot)
    compute_energies(self, dm0_ref, dm0, dm1)

    """
    def __init__(self, emb_pot):
        """Wrapper for PySCF methods.

        Parameters
        ----------
        frag_args : dict
            Parameters for individual fragments:
            molecule, method, basis, xc_code, etc.
        emb_args : dict
            Parameters for the embedding calculation:
            method, basis, x_func, c_func, t_func.

        """
        self.emb_pot = emb_pot

    def compute_energies(self, dm0_ref, dm0, dm1):
        """Save information after embedding calculation."""
        # Get electrostatics
        self.energy_dict["rho0_rho1"] = np.einsum('ab,ba', self.emb_pot.vemb_dict["v_coulomb"], dm0)
        self.energy_dict["nuc0_rho1"] = np.einsum('ab,ba', self.emb_pot.vemb_dict["v0_nuc1"], dm0)
        self.energy_dict["nuc1_rho0"] = np.einsum('ab,ba', self.emb_pot.vemb_dict["v1_nuc0"], dm1)
        self.energy_dict["nuc0_nuc1"] = compute_nuclear_repulsion(self.emb_pot.mol0, self.emb_pot.mol1)
        # Final density functionals
        self.pot_object.assign_dm(0, dm0)
        final_vnad = self.emb_pot.compute_nad_potential()
        self.energy_dict["exc_nad_final"] = final_vnad[0]
        self.energy_dict["et_nad_final"] = final_vnad[1]
        self.vemb_dict["v_nad_xc_final"] = final_vnad[2]
        self.vemb_dict["v_nad_t_final"] = final_vnad[3]
        int_ref_xc = np.einsum('ab,ba', self.emb_pot.vemb_dict["v_nad_xc"], dm0_ref)
        int_ref_t = np.einsum('ab,ba', self.emb_pot.vemb_dict["v_nad_t"], dm0_ref)
        self.energy_dict["int_final_xc"] = np.einsum('ab,ba', self.emb_pot.vemb_dict["v_nad_xc_final"], dm0_ref)
        self.energy_dict["int_final_t"] = np.einsum('ab,ba', self.emb_pot.vemb_dict["v_nad_t_final"], dm0_ref)
        # Linearization terms
        int_emb_xc = np.einsum('ab,ba', self.emb_pot.vemb_dict["v_nad_xc"], dm0)
        int_emb_t = np.einsum('ab,ba', self.emb_pot.vemb_dict["v_nad_t"], dm0)
        self.energy_dict["int_ref_xc"] = int_ref_xc
        self.energy_dict["int_ref_t"] = int_ref_t
        self.energy_dict["int_emb_xc"] = int_emb_xc
        self.energy_dict["int_emb_t"] = int_emb_t
        self.energy_dict["deltalin"] = (int_emb_xc - int_ref_xc) + (int_emb_t - int_ref_t)

    def export_info(to_csv=False, to_h5=False):
        """Export information.
        """
        if to_csv:
            csv_file = 'fdet_energies.csv'
            try:
                with open(csv_file, 'w') as csvfile:
                    csv_columns = [key for key in self.energy_dict]
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    writer.writeheader()
                    for data in self.energy_dict:
                        writer.writerow(data)
            except IOError:
                raise IOError("File %s could not be opened" % csv_file)
        if to_h5:
            h5_file = 'fdet_energies.h5'
            save_dict_to_hdf5(self.energy_dict, h5_file)
        else:
            # Printing to screen
            log.to_screen(self.energy_dict)


def save_dict_to_hdf5(dic, filename):
    """Save dictionary to hdf5 file.

    Parameters
    ----------
    dic : dictionary
        Data to be stored.
    filename : str
        Proper path and name of the file where the data will be saved.
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """Function used for deep dictionary structures.

    Parameters
    ----------
    h5file : h5py.File
        File generated with h5py where things will be stored.
    path : str
        Separation to create the deeper layers.
    dict : dictionary
        Information to be saved

    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))
