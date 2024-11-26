import numpy as np
import tensorflow as tf

rand = tf.random.uniform([1,2 ], 0, 1)
print(tf.rank(rand))
print(rand)