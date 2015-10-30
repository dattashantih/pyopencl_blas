from __future__ import print_function

import numpy as np
import pyopencl as cl
from pyopencl.array import Array
import pyopencl_blas as blas

# start up OpenCL
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# start up the BLAS
blas.setup()

# generate some random data on the CPU
n = 5
dtype = 'float64'  # also supports 'float64'

A = np.zeros((n, n), dtype=dtype)
x = np.zeros(n, dtype=dtype)
x1 = np.zeros(n, dtype=dtype)
x2 = np.zeros(n, dtype=dtype)

rng = np.random.RandomState(1)  # change the seed to see different data
A[...] = rng.uniform(-1, 1, size=A.shape)
x[...] = rng.uniform(-1, 1, size=x.shape)
x1[...] = rng.uniform(-1, 1, size=x1.shape)
x2[...] = rng.uniform(-1, 1, size=x2.shape)

A_upper = np.triu(A)
A = np.tril(A)

# allocate OpenCL memory on the device
clA = Array(queue, A.shape, A.dtype)
clA_upper = Array(queue, A.shape, A.dtype)
clx = Array(queue, x.shape, x.dtype)
clx1 = Array(queue, x1.shape, x1.dtype)
clx2 = Array(queue, x2.shape, x2.dtype)

# copy data to device
clA.set(A)
clA_upper.set(A_upper)
clx.set(x)

# compute a triangular solve (trsv)
blas.trsv(queue, clA, clx)

# check the result
print("Expected: ", np.linalg.solve(A, x))
print("Actual:   ", clx.get())
print()

# try a triangular solve with the transpose
clx1.set(x1)
blas.trsv(queue, clA, clx1, transA=True)
print("Expected: ", np.linalg.solve(A.T, x1))
print("Actual:   ", clx1.get())
print()

# trye an upper triangular solve
clx2.set(x2)
blas.trsv(queue, clA_upper, clx2, lower=False)
print("Expected: ", np.linalg.solve(A_upper, x2))
print("Actual:   ", clx2.get())
print()


# tidy up the BLAS
blas.teardown()
