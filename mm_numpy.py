import numpy as np

# Initialize 2000x2000 matrices with random float32 values
A = np.random.randn(2000, 2000).astype(np.float32)
B = np.random.randn(2000, 2000).astype(np.float32)

# Perform matrix multiplication using numpy.dot()
C = np.dot(A, B)

# Calculate the sum of elements in the result matrix C
sum_C = np.sum(C)

# Print the sum
print("Sum of elements in result matrix C:", sum_C)
