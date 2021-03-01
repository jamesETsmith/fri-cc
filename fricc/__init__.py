import numpy as np

from pyscf.cc import rintermediates as imd
from .py_rccsd import update_amps as my_update_amps


def update_amps_wrapper(t1, t2, eris):
    nocc, nvirt = t1.shape
    t1_new = np.zeros_like(t1, order="C")
    t2_new = np.zeros_like(t2, order="C")

    fock = eris.fock
    fov = np.ascontiguousarray(fock[:nocc, nocc:])
    foo = np.ascontiguousarray(fock[:nocc, :nocc])
    fvv = np.ascontiguousarray(fock[nocc:, nocc:])

    my_update_amps(
        t1,
        t2.ravel(),
        t1_new,
        t2_new.ravel(),
        foo.copy(),
        fov.copy(),
        fvv.copy(),
        eris.oooo.ravel(),
        eris.ovoo.ravel(),
        eris.oovv.ravel(),
        eris.ovvo.ravel(),
        eris.ovov.ravel(),
        eris.get_ovvv().ravel(),
        imd._get_vvvv(eris).ravel(),
        np.ascontiguousarray(eris.mo_energy),
    )

    return t1_new, t2_new