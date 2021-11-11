import numpy as np
from emcee.autocorr import integrated_time

from .friccsd import FRICCSD
from .frilccsd import FRILCCSD


def mcmc_std(vals: np.ndarray) -> list:
    tau_f = integrated_time(vals)[0]
    # print(tau_f, vals.size, np.var(vals))
    return np.sqrt(tau_f / vals.size * np.var(vals)), tau_f
