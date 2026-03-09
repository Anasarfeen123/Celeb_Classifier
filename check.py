import tensorflow as tf

print("GPUs:", tf.config.list_physical_devices("GPU"))

a = tf.random.normal([1000, 1000])
b = tf.random.normal([1000, 1000])
c = tf.matmul(a, b)

print(c)