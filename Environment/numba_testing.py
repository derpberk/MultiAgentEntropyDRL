import numba
import numpy as np
import time

from numba import jit, cfunc, njit
import numba


def foo(M):
	eig = 0
	for i in range(100):
		eig += np.linalg.det(np.abs(M))
	return np.sum(eig)


@jit(numba.float32(numba.float32[:, :]), parallel=True)
def numba_foo(M):
	eig = 0
	for i in range(100):
		eig += np.linalg.det(np.abs(M))
	return np.sum(eig)


N = 10
m = np.random.rand(100,100).astype(np.float32) + 10
print("--- NO NUMBA ---")
t0 = time.time()
b = [foo(m) for _ in range(N)]
print("Result: ", np.sum(b))
print("Calc. Time: ", (time.time()-t0)/N)

print("--- WITH NUMBA ---")
print("Result in first compilation: ", numba_foo(m))

print("--- WITH NUMBA ---")
t0 = time.time()
a = [numba_foo(m) for _ in range(N)]
print("Result: ", np.sum(a))
print("Calc. Time: ", (time.time()-t0)/N)

