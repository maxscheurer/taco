"""PySCF utilities."""

from pyscf import gto


def get_pyscf_molecule(mol, basis):
    """Generate PySCF molecule object.

    Parameters
    ----------
    mol : qcelemental.models.Molecule
        The molecule object.

    """
    string = mol.to_string()
    pyscf_mol = gto.M(
            atom=string,
            basis=basis,)
    return pyscf_mol
