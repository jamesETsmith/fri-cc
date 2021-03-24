import numpy as np
import pytest
from pyscf import gto, scf, cc
from pyscf.cc import rintermediates as imd

from fricc import update_amps_wrapper

#
# Error tolerances
#

ENERGY_TOL = 1e-13
T1_TOL = 1e-13
T2_TOL = 1e-14

#
# Helpers
#


#
# Data

atoms = [
    "H 0 -1 -1; O 0 0 0; H 0 1.2 -1;",
    "xyz_files/methane.xyz",
    "xyz_files/ethane.xyz",
]

n_atoms = len(atoms)

data_small = {"atom": atoms, "basis": ["3-21g"] * n_atoms}
data_medium = {"atom": atoms, "basis": ["augccpvdz"] * n_atoms}
data_large = {"atom": atoms, "basis": ["augccpvtz"] * n_atoms}


#
# Tests
#


@pytest.mark.parametrize(
    "atom,basis",
    [
        (data_small["atom"][i], data_small["basis"][i])
        for i in range(len(data_small["atom"]))
    ],
)
def test_small(atom, basis):
    mol = gto.M(atom=atom, basis=basis)
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf)
    mycc.kernel()

    # unpack them
    pyscf_t1, pyscf_t2 = (mycc.t1, mycc.t2)

    mycc_2 = cc.CCSD(mf)
    mycc_2.update_amps = update_amps_wrapper
    mycc_2.kernel()

    my_t1, my_t2 = (mycc_2.t1, mycc_2.t2)

    # Test energy
    assert abs(mycc.e_tot - mycc_2.e_tot) < ENERGY_TOL
    assert np.linalg.norm(my_t1 - pyscf_t1) < T1_TOL
    assert np.linalg.norm(my_t2 - pyscf_t2) < T2_TOL


# @pytest.mark.parametrize("atom,basis", [(data_medium["atom"][i],data_medium["basis"][i]) for i in range(len(data_medium["atom"]))])
# def test_medium(atom, basis):
#     mol = gto.M(atom=atom, basis=basis)
#     mf = scf.RHF(mol).run()
#     mycc = cc.CCSD(mf)
#     mycc.kernel()

#     # unpack them
#     pyscf_t1, pyscf_t2 = (mycc.t1, mycc.t2)

#     mycc_2 = cc.CCSD(mf)
#     mycc_2.update_amps = update_amps_wrapper
#     mycc_2.kernel()

#     my_t1, my_t2 = (mycc_2.t1, mycc_2.t2)

#     # Test energy
#     assert abs(mycc.e_tot - mycc_2.e_tot) < ENERGY_TOL
#     assert np.linalg.norm(my_t1-pyscf_t1) < T1_TOL
#     assert np.linalg.norm(my_t2-pyscf_t2) < T2_TOL