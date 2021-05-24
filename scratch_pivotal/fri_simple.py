import time
import numpy as np

#
#
#


def fri_compress_vector_slower(v_sparse: np.ndarray, m: int) -> np.ndarray:

    # Shorthand variables
    v_idx = v_sparse[:, 0]
    # Need the copy b/c you start zeroing elements in your original vector
    v = np.copy(v_sparse[:, 1])
    v_abs = np.abs(v)

    # Time
    t_setup = time.time()

    # Setting things up
    tmv = 0  # \tau_v^m (index of compressed vector)
    V = np.zeros((m, 2))  # Compressed vector indices and values
    r = np.linalg.norm(v, ord=1) / m
    s = np.zeros(m, dtype=int)
    s[tmv] = np.argmax(v_abs)
    remaining_norm = np.linalg.norm(v, ord=1)

    # Timing
    t_setup = time.time() - t_setup
    t_exact = time.time()

    # print(v[s[tmv]], r)
    # Find the largest elements and save them exactly
    while v_abs[s[tmv]] >= r:

        # Set our compressed index and value for element tmv
        V[tmv] = [v_idx[s[tmv]], v[s[tmv]]]

        # Update the remaining norm and recalculate threshold to stop searching for exact values
        remaining_norm -= np.abs(V[tmv, 1])
        r = remaining_norm / (m - tmv)

        # Zero the chosen element
        v_abs[s[tmv]] = 0

        # Update index
        tmv += 1

        # Stopping condition
        if tmv >= m:
            break

        # Find the next largest element
        s[tmv] = np.argmax(v_abs)

    if tmv == 0:
        print("NO ELEMENTS KEPT EXACTLY")
    elif tmv == m:
        print("All {} elements are EXACT".format(m))

    # Timing
    t_exact = time.time() - t_exact
    t_sample = time.time()

    # Sample the remaining number of non-zero entries
    if tmv < m and remaining_norm > 1e-10:
        # Sample indices using systematic sampling
        remaining_elements = m - tmv
        sample_r = np.random.uniform(high=1.0 / remaining_elements)
        uk = np.linspace(0, 1, num=remaining_elements, endpoint=False) + sample_r
        intervals = np.append(0, np.cumsum(v_abs / remaining_norm))

        nnz = v.shape[0]
        # interval_idx = np.linspace(0, nnz - 1, num=nnz, dtype=int)
        interval_idx = np.arange(nnz)
        interval_idx.shape = (nnz, 1)
        Nj = np.sum(
            (intervals[interval_idx] <= uk) & (uk < intervals[interval_idx + 1]), 1
        )

        # Calculate the values of remaining elements in the compressed vector
        norm_factor = remaining_norm / remaining_elements
        sample_idx = np.nonzero(Nj)[0]
        print(tmv, sample_idx.shape, v_idx.shape, Nj.shape, nnz, uk.shape)
        # V[tmv:, 0] = v_idx[sample_idx]
        # V[tmv:, 1] = Nj[sample_idx] * np.sign(v[sample_idx]) * norm_factor
        n_sample = sample_idx.size
        V[tmv : n_sample + tmv, 0] = v_idx[sample_idx]
        V[tmv : n_sample + tmv, 1] = (
            Nj[sample_idx] * np.sign(v[sample_idx]) * norm_factor
        )

    else:
        print("Remaining norm = {}".format(remaining_norm))
        print("Only need {} of the requested {} elements".format(tmv, m))
        V = V[:tmv]

    t_sample = time.time() - t_sample

    print(
        "Setup Time = {:.2e}   Large Mag. Time {:.2e}    Sample Time {:.2e}".format(
            t_setup, t_exact, t_sample
        )
    )

    return V


def fri_compress_vector(v_sparse: np.ndarray, m: int) -> np.ndarray:

    # Shorthand variables
    v_idx = v_sparse[:, 0]
    # Need the copy b/c you start zeroing elements in your original vector
    v = np.copy(v_sparse[:, 1])
    v_abs = np.abs(v)

    # Sort for speed (maybe)
    t0 = time.time()
    v_abs_sorted_idx = np.argsort(v_abs)[::-1][:m]
    # print(v_abs[v_abs_sorted_idx[:4]])
    t_sort = time.time() - t0

    # Time
    t_setup = time.time()

    # Setting things up
    tmv = 0  # \tau_v^m (index of compressed vector)
    V = np.zeros((m, 2))  # Compressed vector indices and values
    s = np.zeros(m, dtype=int)
    s[tmv] = v_abs_sorted_idx[0]
    remaining_norm = np.linalg.norm(v, ord=1)
    r = remaining_norm / m

    # Timing
    t_setup = time.time() - t_setup
    t_exact = time.time()

    # print(v[s[tmv]], r)
    # Find the largest elements and save them exactly
    while v_abs[s[tmv]] >= r:

        # Set our compressed index and value for element tmv
        V[tmv] = [v_idx[s[tmv]], v[s[tmv]]]

        # Update the remaining norm and recalculate threshold to stop searching for exact values
        remaining_norm -= np.abs(V[tmv, 1])
        r = remaining_norm / (m - tmv)

        # Zero the chosen element
        v_abs[s[tmv]] = 0

        # Update index
        tmv += 1

        # Stopping condition
        if tmv >= m:
            break

        # Find the next largest element
        s[tmv] = v_abs_sorted_idx[tmv]

    if tmv == 0:
        print("NO ELEMENTS KEPT EXACTLY")
    elif tmv == m:
        print("All {} elements are EXACT".format(m))

    # Timing
    t_exact = time.time() - t_exact
    t_sample = time.time()

    # Sample the remaining number of non-zero entries
    times = {}
    if tmv < m and remaining_norm > 1e-10:
        # Sample indices using systematic sampling
        remaining_elements = m - tmv
        sample_r = np.random.uniform(high=1.0 / remaining_elements)

        t_tmp = time.time()
        uk = np.linspace(0, 1, num=remaining_elements, endpoint=False) + sample_r
        times["t1"] = time.time() - t_tmp

        t_tmp = time.time()
        intervals = np.append(0, np.cumsum(v_abs / remaining_norm))
        times["t2"] = time.time() - t_tmp

        nnz = v.shape[0]
        # interval_idx = np.linspace(0, nnz - 1, num=nnz, dtype=int)
        t_tmp = time.time()
        interval_idx = np.arange(nnz)
        interval_idx.shape = (nnz, 1)
        Nj = np.sum(
            (intervals[interval_idx] <= uk) & (uk < intervals[interval_idx + 1]), 1
        )

        # Calculate the values of remaining elements in the compressed vector
        norm_factor = remaining_norm / remaining_elements
        sample_idx = np.nonzero(Nj)[0]
        print(tmv, sample_idx.shape, v_idx.shape, Nj.shape, nnz, uk.shape)
        # V[tmv:, 0] = v_idx[sample_idx]
        # V[tmv:, 1] = Nj[sample_idx] * np.sign(v[sample_idx]) * norm_factor
        times["t3"] = time.time() - t_tmp

        t_tmp = time.time()
        n_sample = sample_idx.size
        V[tmv : n_sample + tmv, 0] = v_idx[sample_idx]
        V[tmv : n_sample + tmv, 1] = (
            Nj[sample_idx] * np.sign(v[sample_idx]) * norm_factor
        )
        times["t4"] = time.time() - t_tmp

        print("{:16s} {:.2e}".format("Linespace time", times["t1"]))
        print("{:16s} {:.2e}".format("Intervals time", times["t2"]))
        print("{:16s} {:.2e}".format("Nj sum time", times["t3"]))
        print("{:16s} {:.2e}".format("Setting time", times["t4"]))
    else:
        print("Remaining norm = {}".format(remaining_norm))
        print("Only need {} of the requested {} elements".format(tmv, m))
        V = V[:tmv]

    t_sample = time.time() - t_sample

    print(
        "Setup Time = {:.2e}   Large Mag. Time {:.2e}    Sample Time {:.2e}".format(
            t_setup, t_exact + t_sort, t_sample
        )
    )

    return V


def MspV(A: np.ndarray, spv: np.ndarray) -> np.ndarray:

    nrows = A.shape[0]
    result = np.zeros(nrows)

    for r, _ in enumerate(result):
        for c, val in spv:
            result[r] += val * A[r, int(c)]

    return result


def decompress_vector(spv: np.ndarray, n: int) -> np.ndarray:

    dv = np.zeros(n)

    for i, val in spv:
        dv[int(i)] = val

    return dv


def convert_to_sparse_vector(tensor: np.ndarray) -> np.ndarray:

    n_elements = tensor.size
    result = np.zeros((n_elements, 2))
    result[:, 0] = np.arange(n_elements)
    result[:, 1] = tensor.flatten()
    return result
