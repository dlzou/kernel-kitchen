import numpy as np
import warp as wp

from utils import get_device

@wp.kernel
def dot_kernel(a: wp.array(dtype=wp.vec3),
                  b: wp.array(dtype=wp.vec3),
                  c: wp.array(dtype=float)):

    # get thread index
    tid = wp.tid()

    # load two vec3s
    x = a[tid]
    y = b[tid]

    # compute the dot product between vectors
    r = wp.dot(x, y)

    # write result back to memory
    c[tid] = r


if __name__ == "__main__":
    a = wp.array(np.random.randn(1024, 3), dtype=wp.vec3)
    b = wp.array(np.random.randn(1024, 3), dtype=wp.vec3)
    c = wp.zeros(1024, dtype=float)

    wp.launch(kernel=dot_kernel, # kernel to launch
          dim=1024,             # number of threads
          inputs=[a, b, c],     # parameters
          device="cuda")        # execution device

    print(c)
