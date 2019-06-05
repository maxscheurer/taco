import numpy as np

import psi4
import IPython


def build_supersystem(mol1, mol2):
    # geom, mass, elem, elez, uniq
    geom1, _, elem1, _, _ = mol1.to_arrays()
    geom2, _, elem2, _, _ = mol2.to_arrays()
    geom_merged = np.vstack((geom1, geom2))
    elem_merged = np.append(elem1, elem2).reshape(geom_merged.shape[0], 1)
    mol = np.hstack((elem_merged, geom_merged))
    mol_string = ""
    for a in mol:
        astr = " ".join([str(en) for en in a])
        mol_string += astr + "\n"
    temp = """
    0 1
    {mol}
    units au
    symmetry c1
    no_reorient
    no_com
    """.format(mol=mol_string)
    return psi4.geometry(temp)

h2o = psi4.geometry("""
0 1
O  -7.9563726699    1.4854060709    0.1167920007
H  -6.9923165534    1.4211335985    0.1774706091
H  -8.1058463545    2.4422204631    0.1115993752
symmetry c1
no_reorient
no_com
""")

co = psi4.geometry("""
0 1
C  -3.6180905689    1.3768035675   -0.0207958979
O  -4.7356838533    1.5255563000    0.1150239130
symmetry c1
no_reorient
no_com
""")

psi4.core.be_quiet()

psi4.set_options({'basis': 'cc-pvdz'})
e_a, wfn_a = psi4.energy("SCF", molecule=co, return_wfn=True)
bas_a = wfn_a.basisset()
psi4.core.clean()

psi4.set_options({'basis': 'cc-pvdz'})
e_b, wfn_b = psi4.energy("SCF", molecule=h2o, return_wfn=True)
dm_tot_b = wfn_b.Da().np + wfn_b.Db().np
bas_b = wfn_b.basisset()

mints = psi4.core.MintsHelper(wfn_a)
eri = mints.ao_eri(bas_b, bas_b, bas_a, bas_a)
v_j = np.einsum('ab,abcd->cd', dm_tot_b, eri)
# print(v_j)

nuc_potential_b = psi4.core.ExternalPotential()
for i in range(h2o.natom()):
    geom = np.array([h2o.x(i), h2o.y(i), h2o.z(i)])
    if h2o.units() == 'Angstrom':
        geom *= psi4.constants.bohr2angstroms
    nuc_potential_b.addCharge(h2o.Z(i), *geom)
v_b = nuc_potential_b.computePotentialMatrix(bas_a)
print(v_b.np)

system = build_supersystem(co, h2o)


# grid stuff
basis_obj = psi4.core.BasisSet.build(co, 'ORBITAL', "cc-pvdz")


grid = psi4.core.DFTGrid.build(co, basis_obj)


slater = psi4.core.LibXCFunctional("LDA_X", True)
vwn = psi4.core.LibXCFunctional("LDA_C_VWN", True)
superfunc = psi4.core.SuperFunctional()
superfunc.add_x_functional(slater)
superfunc.add_c_functional(vwn)


Vpot = psi4.core.VBase.build(basis_obj, superfunc, "RV")
Vpot.initialize()
points_func = Vpot.properties()[0]
D = np.array(wfn_a.Da())

IPython.embed()

xc_e = 0
for b in range(len(grid.blocks())):
    # Obtain block information
    block = grid.blocks()[b]
    points_func.compute_points(block)
    npoints = block.npoints()
    lpos = np.array(block.functions_local_to_global())
    # Obtain the grid weight
    w = np.array(block.w())

    # Compute phi!
    phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

    # Build a local slice of D
    lD = D[(lpos[:, None], lpos)]

    # Copmute rho
    rho = 2.0 * np.einsum('pm,mn,pn->p', phi, lD, phi)

    inp = {}
    inp["RHO_A"] = psi4.core.Vector.from_array(rho)

    # Compute the kernel
    ret = superfunc.compute_functional(inp, -1)

    # Compute the XC energy
    vk = np.array(ret["V"])[:npoints]
    xc_e += np.einsum('a,a->', w, vk)

    # Compute the XC derivative.
    v_rho_a = np.array(ret["V_RHO_A"])[:npoints]
    Vtmp = np.einsum('pb,p,p,pa->ab', phi, v_rho_a, w, phi)

IPython.embed()
