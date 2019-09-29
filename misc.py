import tensorflow as tf
import numpy as np
import numpy.random as random

A = random.randint(0, 10, (3, 4))
print(A)
A_centered = A - np.mean(A, axis=0)
print(A_centered)