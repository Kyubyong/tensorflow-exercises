import tensorflow as tf

def evaluate(x, y, z, x_value, y_value):
    """
    x: A placeholder for a scalar Tensor
    y: A placeholder for a scalar Tensor
    z: A Tensor involving x and y
    x_value: A numpy value
    y_value: A numpy value

    Returns the value of `z` when x_value is substituted for x
     and y_value is substituted for y
    """
    with tf.Session() as sess:
        out = sess.run(z, {x: x_value, y: y_value})

    return out


if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [])
    y = tf.placeholder(tf.float32, [])
    z = x + y
    assert evaluate(x, y, z, 1, 2) == 3
    print "SUCCESS!"
