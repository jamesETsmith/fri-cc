import os
import pandas as pd
from pyscf import gto, scf, cc
from py_rccsd import update_amps as my_update_amps

#
# User settings
#

for d in ["_data", "_logs"]:
    os.makedirs(d, exist_ok=True)

#
##
