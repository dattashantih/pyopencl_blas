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

x = np.zeros(n, dtype=dtype)
y = np.zeros(n, dtype=dtype)

rng = np.random.RandomState(1)  # change the seed to see different data
x[...] = rng.uniform(-1, 1, size=x.shape)
y[...] = rng.uniform(-1, 1, size=y.shape)

# allocate OpenCL memory on the device
clx = Array(queue, x.shape, x.dtype)
cly = Array(queue, y.shape, y.dtype)
cld = Array(queue, 1, x.dtype)

# copy data to device
clx.set(x)
cly.set(y)

# compute a dot product (dot)
blas.dot(queue, clx, cly, cld)

# check the result
print("Expected: ", np.dot(x,y))
print("Actual:   ", cld.get()[0])

# tidy up the BLAS
blas.teardown()
