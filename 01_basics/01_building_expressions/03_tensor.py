# Fill in the TODOs in this exercise, then run
# python 03_tensor.py to see if your solution works!
import numpy as np
import tensorflow as tf

def make_tensor(rank, dtype=tf.float32):
    """
    Returns a placeholder for a tensor with rank of `rank`
    rank: the total number of dimensions of the tensor.
    """

    raise NotImplementedError("TODO: implement this function.")

def broadcasted_add(a, b):
    """
    a: a 3D tensor
    b: a 4D tensor
    Returns c, a 4D tensor, where

    c[i, j, k, l] = a[l, k, i] + b[i, j, k, l]

    for all i, j, k, l
    """

    raise NotImplementedError("TODO: implement this function.")

def partial_max(a):
    """
    a: a 4D tensor

    Returns b, a 2-D tensor, where

    b[i, j] = max_{k,l} a[i, k, l, j]

    for all i, j
    """

    raise NotImplementedError("TODO: implement this function.")


if __name__ == "__main__":
    a = make_tensor(3)
    b = make_tensor(4)
    c = broadcasted_add(a, b)
    d = partial_max(c)

    actual = d

    rng = np.random.RandomState([1, 2, 3])
    a_value = rng.randn(2, 2, 2).astype(np.float32)
    b_value = rng.rand(2, 2, 2, 2).astype(np.float32)
    c_value = np.transpose(a_value, (2, 1, 0))[:, None, :, :] + b_value
    expected = c_value.max(axis=1).max(axis=1)

    with tf.Session() as sess:
        actual = sess.run(actual, {a:a_value, b: b_value, c:c_value})
        assert np.allclose(actual, expected)
        print "SUCCESS!"

