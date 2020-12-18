import numpy as np


def _common_data(arr):
    return arr.shape[0], int(np.log2(arr.shape[0]))


def dft(arr):
    n, _ = _common_data(arr)
    j = np.arange(n)
    k = j.reshape((n, 1))

    return np.exp((-2j * np.pi / n) * k * j) @ arr  # python: 2j = 2*sqrt(-1) = 2i


def recursive_fft(arr: np.ndarray):
    n, _ = _common_data(arr)

    if n == 1:
        return arr

    y_even = recursive_fft(arr[::2])
    y_odd = recursive_fft(arr[1::2])

    w = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    tmp = w * y_odd

    return np.concatenate([y_even + tmp, y_even - tmp])


def _bit_reverse_numbers(arr, log_n):
    rev = 0
    for _ in range(log_n):
        rev = (rev << 1) + (arr & 1)
        arr >>= 1
    return rev


def _bit_reverse(arr, n, log_n):
    b = np.empty(n, dtype=complex)
    new_indices = _bit_reverse_numbers(np.arange(n), log_n)
    b[new_indices] = arr
    return b


def iterative_fft(arr):
    n, log_n = _common_data(arr)

    if n < 2:
        return arr

    arr = _bit_reverse(arr, n, log_n)
    indices = np.expand_dims(np.arange(n), -1)

    mm = 2 ** np.arange(1, log_n + 1)
    w_mm = np.exp(-2j * np.pi / mm)
    slice_whole = np.arange(mm.max() // 2)

    for m, w_m in zip(mm, w_mm):
        slice_ = slice_whole[:m//2]

        even_indices = np.concatenate(indices[::m] + slice_)
        odd_indices = np.concatenate(indices[m//2::m] + slice_)

        tmp = (arr[odd_indices].reshape((-1, m//2)) * w_m ** range(m//2)).reshape(-1)

        arr[odd_indices] = arr[even_indices] - tmp
        arr[even_indices] = arr[even_indices] + tmp

    return arr