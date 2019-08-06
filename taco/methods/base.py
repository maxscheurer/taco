"""
FDE Methods Base Class.
"""


class BaseMethod():
    """Base class for method objects.

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
        Perform the actual calculation.

    """
    def __init__(self, mol):
        """ BaseMethod object.

        Parameters
        ----------
        mol : qcelemental Molecule object
            Molecule information
        """
        self.mol = mol
        self.new = True
        self.density = []
        self.energy = {}

    @property
    def restricted(self):
        """Whether it is Restricted case."""
        return self.mol.multiplicity == 1

    def get_density(self):
        """Return density/densities."""
        raise NotImplementedError

    def get_energy(self):
        """Return energy/energies."""
        raise NotImplementedError
