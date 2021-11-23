import pytest
import numpy as np
from pyscf import gto, scf, cc
import fricc

npt = np.testing

#
# All answers from NWChem (example input below)
#
# start methanol
# title "Methanol LCCSD"
# 
# permanent_dir ./scratch
# scratch_dir ./scratch
# 
# memory 50 GB
# 
# 
#  geometry
#    H      1.2001      0.0363      0.8431
#    C      0.7031      0.0083     -0.1305
#    H      0.9877      0.8943     -0.7114
#    H      1.0155     -0.8918     -0.6742
#    O     -0.6582     -0.0067      0.1730
#    H     -1.1326     -0.0311     -0.6482
#  end
#  basis
#     * library 6-31g
#  end
#  task scf
#  task ccsd
#  TCE
#   SCF
#   DIIS 0
#   LCCSD
#   THRESH 1e-10
#  END
#  TASK TCE ENERGY

#
# Data dictionaries
#
atoms = {
    "benzene": """  H      1.2194     -0.1652      2.1600
  C      0.6825     -0.0924      1.2087
  C     -0.7075     -0.0352      1.1973
  H     -1.2644     -0.0630      2.1393
  C     -1.3898      0.0572     -0.0114
  H     -2.4836      0.1021     -0.0204
  C     -0.6824      0.0925     -1.2088
  H     -1.2194      0.1652     -2.1599
  C      0.7075      0.0352     -1.1973
  H      1.2641      0.0628     -2.1395
  C      1.3899     -0.0572      0.0114
  H      2.4836     -0.1022      0.0205""",
    "nitrogen": "N 0 0 0; N 0 0 1.074",
    "methanol": """  H      1.2001      0.0363      0.8431
  C      0.7031      0.0083     -0.1305
  H      0.9877      0.8943     -0.7114
  H      1.0155     -0.8918     -0.6742
  O     -0.6582     -0.0067      0.1730
  H     -1.1326     -0.0311     -0.6482""",
}
nwchem = {
    "benzene": {
        "scf": -230.622518244261,
        "ccsd": -231.189597576392543,
        "lccd": -231.202160668287888,
        "lccsd": -231.208193819574149,
    },
    "nitrogen": {
        "scf": -108.867211348323,
        "ccsd": -109.089712759174020,
        "lccd": -109.090826370341873,
        "lccsd": -109.094095786363752,
    },
    "methanol": {
        "scf": -114.985033959273,
        "ccsd": -115.220670875544798,
        "lccd": -115.221497341492167,
        "lccsd": -115.223286488470634,
    },
}

#
# Tests
#
@pytest.mark.parametrize("system", [
    # ("benzene"),
    ("nitrogen"), 
    ("methanol"),
    ])
def test_lincc(system):
    mol = gto.M(atom=atoms[system], basis="6-31g", verbose=4)

    mf = scf.RHF(mol).run()
    # exit(0)
    scf_error = abs(mf.e_tot - nwchem[system]["scf"])
    print(f"Error in SCF {scf_error:.3e} (Ha)")

    myccsd = cc.rccsd.RCCSD(mf).run()
    ccsd_error = abs(myccsd.e_tot - nwchem[system]["ccsd"])
    print(f"Error in CCSD {ccsd_error:.3e} (Ha)")

    #
    # LCCD
    #
    mylccd = fricc.FRILCCSD(
        mf,
        fri_settings={
            "compressed_contractions": [],
            "compression": "largest",
            "m_keep": 10,
            "LCCD": True,
        },
    )
    mylccd.kernel()
    lccd_error = abs(mylccd.e_tot - nwchem[system]["lccd"])
    print(f"Error in LCCD {lccd_error:.3e} (Ha)")
    npt.assert_almost_equal(mylccd.e_tot, nwchem[system]["lccd"], decimal=6)

    #
    # LCCSD
    #

    mylccsd = fricc.FRILCCSD(
        mf,
        fri_settings={
            "compressed_contractions": [],
            "compression": "largest",
            "m_keep": 10,
        },
    )
    # mylccsd.frozen = 2
    mylccsd.kernel()

    lccsd_error = abs(mylccsd.e_tot - nwchem[system]["lccsd"])
    print(f"Error in LCCSD {lccsd_error:.3e} (Ha)")
    npt.assert_almost_equal(mylccsd.e_tot, nwchem[system]["lccsd"], decimal=6)
