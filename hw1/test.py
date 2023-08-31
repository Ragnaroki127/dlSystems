import sys
sys.path.append('./apps')
sys.path.append('./python')

import needle as ndl

import numpy as np

def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [x.numpy() for x in out.op.gradient_as_tuple(ndl.Tensor(np.ones(out.shape)), out)]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i])
        for i in range(len(args))
    )
    print(numerical_grads, computed_grads)
    return numerical_grads, computed_grads

if __name__ == "__main__":
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(1, 3)), shape=(2, 3, 3))
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(3, 1)), shape=(3, 3))
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(1, 3)), shape=(3, 3))
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(1,)), shape=(3, 3, 3))
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn()), shape=(3, 3, 3))
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(5,4,1)), shape=(5,4,3))