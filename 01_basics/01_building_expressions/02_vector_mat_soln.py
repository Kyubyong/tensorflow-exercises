import numpy as np
import tensorflow as tf

def make_vector():
    """
    Returns a new 1-D Tensor placeholder of float32.
    """

    return tf.placeholder(tf.float32, [None])

def make_matrix():
    """
    Returns a new 2-D Tensor placeholder of float32.
    """

    return tf.placeholder(tf.float32, [None, None])

def elemwise_mul(a, b):
    """
    a: A 2-D Tensor
    b: A 2-D Tensor
    Returns the elementwise product of a and b
    """

    return tf.mul(a, b)

def matrix_vector_mul(a, b):
    """
    a: A 2-D Tensor
    b: A 1-D Tensor
    Returns the matrix-vector product of a and b
    """

    return tf.squeeze(tf.matmul(a, tf.expand_dims(b, 1)), [1])

if __name__ == "__main__":
    a = make_vector()
    b = make_vector()
    c = elemwise_mul(a, b)
    d = make_matrix()
    e = matrix_vector_mul(d, c)
    actual = e
    rng = np.random.RandomState([1, 2, 3])
    a_value = rng.randn(5).astype(np.float32)
    b_value = rng.rand(5).astype(np.float32)
    c_value = a_value * b_value
    d_value = rng.randn(5, 5).astype(np.float32)
    expected = np.dot(d_value, c_value)
    
    with tf.Session() as sess:
        actual = sess.run(actual, {a:a_value, b: b_value, d:d_value})
        assert np.allclose(actual, expected)
        print "SUCCESS!"
