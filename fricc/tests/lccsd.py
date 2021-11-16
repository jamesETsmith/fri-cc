from pyscf import gto, scf, cc
import fricc

#
#
#
system = "benzene"
# system = "nitrogen"

#
#
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
}
nwchem = {
    "benzene": {
        "scf": -230.622518244261,
        "ccsd": -231.189597576392543,
        "lccsd": -231.208193819574149,
    },
    "nitrogen": {
        "scf": -108.867211348323,
        "ccsd": -109.089712759174020,
        "lccsd": -109.094095786363752,
    },
}

mol = gto.M(atom=atoms[system], basis="6-31g", verbose=4)

mf = scf.RHF(mol).run()
# exit(0)
scf_error = abs(mf.e_tot - nwchem[system]["scf"])
print(f"Error in SCF {scf_error}")

myccsd = cc.CCSD(mf).run()
ccsd_error = abs(myccsd.e_tot - nwchem[system]["ccsd"])
print(f"Error in CCSD {ccsd_error}")

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
print(f"Error in CCSD {lccsd_error}")
