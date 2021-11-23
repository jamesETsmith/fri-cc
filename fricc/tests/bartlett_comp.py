from pyscf import gto, scf, cc
from fricc import fake_lcc

#
#
#
system = "h2"
system = "h-"
# system = "he"

#
#
#
atoms = {
    "h2": "H 0 0 0; H 0 0 0.7430;",
    "h-": "H 0 0 0;",
    "he": "He 0 0 0;",
}
bartett = {
    "h2": {"error": -0.000749},
    "h-": {"error": -0.00256},
    "he": {"error": -0.000322},
}

mol = gto.M(atom=atoms[system], basis="aug-cc-pVTZ", charge = -1 if system == "h-" else 0, verbose=4)

mf = scf.RHF(mol).run()

myccsd = cc.rccsd.RCCSD(mf).run()

#
# LCCSD
#

mylccsd = fake_lcc.LCCSD(mf)
mylccsd.kernel()
lccsd_error = mylccsd.e_tot - myccsd.e_tot
my_error = abs(lccsd_error - bartett[system]["error"])
print(f"Error in LCCSD {lccsd_error:.3e} (Ha)")
print(f"Difference compared to Bartlett {my_error:.3e} (Ha)")
