# Fill in the TODOs in this exercise, then run
# python 01_scalar.py to see if your solution works!
import numpy as np
import tensorflow as tf
raise NotImplementedError("TODO: add any other imports you need")

def make_scalar():
    """
    Returns a new placeholder for a scalar tensor of float32.
    """

    raise NotImplementedError("TODO: implement this function.")

def log(x):
    """
    Returns the logarithm of a scalar tensor.
    """

    raise NotImplementedError("TODO: implement this function.")

def add(x, y):
    """
    Adds two scalar tensors together and returns the result.
    """

    raise NotImplementedError("TODO: implement this function.")

if __name__ == "__main__":
    a = make_scalar()
    b = make_scalar()
    c = log(b)
    d = add(a, c)
    actual = d
    expected = 1. + np.log(2.)
    with tf.Session() as sess:
        actual = sess.run(actual, {a:1., b: 2.})
        assert np.allclose(actual, expected)
        print "SUCCESS!"
