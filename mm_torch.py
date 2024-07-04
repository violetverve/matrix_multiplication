import torch

# Set the device to CPU explicitly
device = torch.device("cpu")

# Set the number of threads to match the number of CPU cores
torch.set_num_threads(4)  # Adjust the number as per your CPU's core count

# Initialize matrices
A = torch.randn(2000, 2000, dtype=torch.float32, device=device)
B = torch.randn(2000, 2000, dtype=torch.float32, device=device)

# Perform matrix multiplication
C = A@B

# Calculate the sum of elements in the result matrix C
sum_C = torch.sum(C).item()

# Print the sum
print("Sum of elements in result matrix C:", sum_C)
