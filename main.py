import numpy as np
import timeit
from fft import dft, iterative_fft, recursive_fft


def assert_close_to_ref(vals, ref):
    assert np.allclose(vals, ref)


if __name__ == '__main__':
    x = np.random.random(2 ** 12)

    funcs = [dft, iterative_fft, recursive_fft]
    f_names = ["dft", "recursive_fft", "iterative_fft"]

    numpy_fft = np.fft.fft(x)
    t = timeit.timeit('np.fft.fft(x)', setup="import numpy as np; from __main__ import x", number=100)
    print(f"Time for numpy fft: {t:.4f}s")

    for f, fname in zip(funcs, f_names):
        t = timeit.timeit(f'{fname}(x)', setup=f"from __main__ import x, {fname}", number=10)
        print(f"Time for {fname}: {t:.4f}s")
        assert np.allclose(f(x), numpy_fft)
