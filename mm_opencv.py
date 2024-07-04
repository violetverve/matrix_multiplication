import cv2
import numpy as np

# Create two 2000x2000 matrices with random float values
A = np.random.rand(2000, 2000).astype(np.float32)
B = np.random.rand(2000, 2000).astype(np.float32)

# Perform matrix multiplication using OpenCV
C = cv2.gemm(A, B, 1, None, 0)

# Calculate and print the sum of elements in the result matrix C
sum_C = np.sum(C)
print("Sum of elements in result matrix C:", sum_C)
