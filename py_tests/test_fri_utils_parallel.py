import numpy as np
from scipy.optimize import curve_fit
from fricc.py_rccsd import parallel_sample
import pytest
import matplotlib.pyplot as plt
import time

np.random.seed(20)


# @pytest.mark.parametrize(
#     "m, n", [(10, 100), (50, 100), (100, 100), (int(1e3), int(1e7))]
# )
# def test_get_m_largest(m, n):
#     v1 = np.random.random(n).astype(np.float64)

#     np_idx = np.flip(np.argsort(v1))[:m]
#     fricc_idx = np.zeros(m, dtype=np.uint64)

#     get_m_largest(v1, m, fricc_idx)
#     np.testing.assert_array_equal(np_idx, fricc_idx)


def find_d_largest(n_sample, x):
    remaining_norm = np.linalg.norm(x, 1)
    D = []
    sort_idx = np.flip(np.argsort(np.abs(x)))

    for i in range(n_sample):
        idx = sort_idx[i]
        d = len(D)
        xi = abs(x[idx])
        if (n_sample - d) * xi >= remaining_norm - xi:
            D.append(idx)
            remaining_norm -= xi
            d -= 1
        else:
            break
    return D, remaining_norm


def make_p(x, n_sample, D, remaining_norm):
    p = np.zeros(x.size)
    indices = [i for i in range(x.size) if i not in D]
    for i in range(len(indices)):
        idx = indices[i]
        p[idx] = abs(x[idx]) * n_sample / remaining_norm
    return p


def convert_sparse_to_dense(idx, vals, size):
    vec = np.zeros(size)
    for i, idx_i in enumerate(idx):
        vec[idx_i] = vals[i]
    return vec


def fake_systematic(n_sample, p):
    sample_r = np.random.rand()
    uk = np.linspace(0, n_sample - 1, num=n_sample, endpoint=True) + sample_r
    # print(uk)
    intervals = np.concatenate(([0], np.cumsum(p)))
    # print(intervals)
    # exit(0)
    last_interval = 1
    S = np.zeros(n_sample, dtype=int)
    for i in range(uk.size):
        for j in range(last_interval, intervals.size - 1):
            if uk[i] >= intervals[j] and uk[i] < intervals[j + 1]:
                S[i] = np.copy(j)
                last_interval = np.copy(j)
                break
    return S.tolist()


vec_size = 10000000
n_sample = 10000

n_iter = 10000
print_step = 1

# Choosing the vector we want to compress
x = np.random.rand(vec_size) + np.random.rand(vec_size) * -1.0
# x = np.ones(vec_size)  # Breaks pivotal

# Error helpers
norm1 = np.linalg.norm(x, 1)
norm2 = np.linalg.norm(x, 2)
x_compare = np.zeros(vec_size)
instant_errors = np.zeros((2, n_iter))
avg_errors = np.zeros((2, n_iter))

D, remaining_norm = find_d_largest(n_sample, x)
D_vals = x[D]
print("Length of D", len(D))

# Make the vector of probabilities
p = make_p(x, n_sample - len(D), D, remaining_norm)
print(f"Sum of p = {np.sum(p)}")
# print(p)

for i in range(n_iter):
    # S = fake_systematic(n_sample - len(D), p)
    t0 = time.time()
    S = parallel_sample(n_sample - len(D), p)
    t_sample = time.time() - t0
    # print(t_sample)
    exit(0)
    # print(S)
    # if i > 1:
    #     exit(0)
    # print(S)
    x_i = convert_sparse_to_dense(D + S, x[D + S] / p[D + S], vec_size)
    x_compare += x_i
    # print(x_i)
    # print(x_compare)

    # Track errors

    instant_errors[0, i] = np.linalg.norm(x - x_i, 1) / norm1
    instant_errors[1, i] = np.linalg.norm(x - x_i, 2) / norm2
    avg_errors[0, i] = np.linalg.norm(x - x_compare / (i + 1), 1) / norm1
    avg_errors[1, i] = np.linalg.norm(x - x_compare / (i + 1), 2) / norm2

    if i % print_step == 0:
        print(
            "Iter. {:d}:\tTime (s):{:.2e}\tInstant L1: {:.4e}\tAvg L1:{:.2e}\tInstant L2: {:.4e}\tAvg L2: {:.2e}".format(
                i,
                t_sample,
                instant_errors[0, i],
                avg_errors[0, i],
                instant_errors[1, i],
                avg_errors[1, i],
            )
        )
    # if i == 2:
    #     exit(0)

# print(x_compare)

#
# Curve Fitting
#

iters = np.arange(n_iter) + 1


def power_law(x: np.array, a: float, b: float) -> np.array:
    return a * np.power(x, b)


popt1, _ = curve_fit(power_law, iters, avg_errors[0, :])
popt2, _ = curve_fit(power_law, iters, avg_errors[1, :])


#
# Plot
#
plt.figure()
plt.loglog(iters, avg_errors[0, :], label="L1 Error", color="tab:blue")
plt.loglog(iters, avg_errors[1, :], label="L2 Error", color="tab:orange")
plt.loglog(
    iters,
    power_law(iters, *popt1),
    label=f"L1 Fit: x^({popt1[1]:0.2f})",
    linestyle="--",
    color="tab:blue",
)
plt.loglog(
    iters,
    power_law(iters, *popt2),
    label=f"L2 Fit: x^({popt2[1]:0.2f})",
    linestyle="--",
    color="tab:orange",
)
# plt.loglog(iters, np.power(iters, -0.5), label="x^(-1/2)")
plt.legend()
plt.savefig("pivotal_parallel.png")
