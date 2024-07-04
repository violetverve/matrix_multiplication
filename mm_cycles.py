#include <iostream>
#include <vector>
#include <thread>

const int SIZE = 2000; // Size of the matrices (2000x2000)
const int NUM_THREADS = 4; // Number of threads to use for parallel computation

// Function to initialize a matrix with random float values (for demonstration)
void initializeMatrix(std::vector<std::vector<float>>& matrix) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            matrix[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}

// Function to transpose a matrix
void transposeMatrix(const std::vector<std::vector<float>>& matrix, std::vector<std::vector<float>>& transposed) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
}

// Function to perform matrix multiplication for a specific range of rows
void multiplyMatrices(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B,
                      std::vector<std::vector<float>>& result, int startRow, int endRow) {
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            float sum = 0.0;
            for (int k = 0; k < SIZE; ++k) {
                sum += A[i][k] * B[j][k]; // Multiply row i of A with row j of B
            }
            result[i][j] = sum;
        }
    }
}

int main() {
    // Initialize matrices A, B, and result
    std::vector<std::vector<float>> A(SIZE, std::vector<float>(SIZE));
    std::vector<std::vector<float>> B(SIZE, std::vector<float>(SIZE));
    std::vector<std::vector<float>> B_transposed(SIZE, std::vector<float>(SIZE));
    std::vector<std::vector<float>> result(SIZE, std::vector<float>(SIZE));

    // Initialize matrices A and B (for demonstration, you can use your own initialization logic)
    initializeMatrix(A);
    initializeMatrix(B);

    // Transpose matrix B
    transposeMatrix(B, B_transposed);

    // Create threads for parallel matrix multiplication
    std::vector<std::thread> threads;
    int rowsPerThread = SIZE / NUM_THREADS;
    for (int t = 0; t < NUM_THREADS; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = (t == NUM_THREADS - 1) ? SIZE : (startRow + rowsPerThread);
        threads.emplace_back(multiplyMatrices, std::ref(A), std::ref(B_transposed), std::ref(result), startRow, endRow);
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Calculate the sum of the resulting matrix for validation
    float sum = 0.0;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            sum += result[i][j];
        }
    }

    // Print the sum of the result matrix
    std::cout << "Sum of result matrix elements: " << sum << std::endl;

    return 0;
}
