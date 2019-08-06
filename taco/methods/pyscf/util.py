"""PySCF utilities."""

from pyscf import gto


def get_pyscf_molecule(mol, basis):
    """Generate PySCF molecule object.

    Parameters
    ----------
    mol : qcelemental.models.Molecule
        The molecule object.

    """
    if not hasattr(mol, 'molecular_multiplicity'):
        raise AttributeError("Molecule must have multiplicity.")
    multiplicity = mol.molecular_multiplicity
    spin = multiplicity - 1
    string = mol.to_string(dtype='xyz')
    lines = string.splitlines()
    count = 0
    for line in lines:
        if len(line.split(' ')) < 4:
            count += 1
    string = '\n'.join(lines[count:])
    pyscf_mol = gto.M(
            atom=string,
            basis=basis,
            spin=spin,)
    return pyscf_mol
