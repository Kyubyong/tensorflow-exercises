import numpy as np
import tensorflow as tf

def make_scalar():
    """
    Returns a new placeholder for a scalar tensor of float32.
    """

    return tf.placeholder(tf.float32, [])

def log(x):
    """
    Returns the logarithm of a scalar tensor.
    """

    return tf.log(x)

def add(x, y):
    """
    Adds two scalar tensors together and returns the result.
    """

    return tf.add(x, y)

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
