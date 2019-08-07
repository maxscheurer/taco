"""
FDE SCF Methods Base Class.
"""

from qcelemental.models import Molecule


class SCFMethod():
    """Base class for SCF method objects.

    Attributes
    ----------
    density : list[np.ndarray(dtype=float)]
        List with densities obtained with this method.
    energy : dict('name', np.ndarray)
        Energies obtained with this method.

    Properties
    ----------
    restricted : bool
        Wheter the wavefunction is restricted or not.

    Methods
    ---------
    __init__(self, mol)
        Initialize the method.
    get_density(self) :
        Return density/densities.
    solve :
        Perform the SCF calculation.
    perturbe_fock(self, pot) :
        Add a potential to the Fock matrix.

    """
    def __init__(self, mol):
        """ SCFMethod object.

        Parameters
        ----------
        mol : qcelemental Molecule object
            Molecule information.
        """
        if not isinstance(mol, Molecule):
            raise TypeError('Molecule object should be instance of qcelemental.models.Molecule')
        self.mol = mol
        self.new = True
        self.density = []
        self.energy = {}

    @property
    def restricted(self):
        """Whether it is Restricted case."""
        return self.mol.molecular_multiplicity == 1

    def get_density(self):
        """Return the DM(s)."""
        if self.density != []:
            return self.density
        else:
            self.solve(conv_tol=1e-12)
            return self.density

    def get_energy(self):
        """Return the SCF energy."""
        if self.energy.get("scf", 0):
            return self.energy["scf"]
        else:
            self.solve(conv_tol=1e-12)
            return self.energy["scf"]

    def get_fock(self):
        """Contruct Fock matrix."""
        raise NotImplementedError

    def perturbe_fock(self, pot):
        """Add an effective potential to the Fock matrix.

        Parameters
        ----------
        pot : np.ndarray(dtype=float)
            Effective potential in the form of a Fock matrix.

        """
        raise NotImplementedError

    def solve(self):
        """Perform SCF calculation."""
        raise NotImplementedError
