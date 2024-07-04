#include <iostream>
#include <vector>
#include <cblas.h>
#include <iomanip>

void initialize_matrix(std::vector<float>& mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void print_matrix(const std::vector<float>& mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(2) << mat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

float sum_matrix(const std::vector<float>& mat, int rows, int cols) {
    float sum = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        sum += mat[i];
    }
    return sum;
}

int main() {
    int rowsA = 2000, colsA = 2000;
    int rowsB = 2000, colsB = 2000;

    std::vector<float> A(rowsA * colsA);
    std::vector<float> B(rowsB * colsB);
    std::vector<float> C(rowsA * colsB, 0);

    // Initialize matrices A and B with random values
    initialize_matrix(A, rowsA, colsA);
    initialize_matrix(B, rowsB, colsB);

    // Perform matrix multiplication using OpenBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rowsA, colsB, colsA,
                1.0f, A.data(), colsA, B.data(), colsB,
                0.0f, C.data(), colsB);

    // Calculate the sum of elements in the result matrix C
    float sum = sum_matrix(C, rowsA, colsB);

    std::cout << "Sum of elements in result matrix C: " << sum << std::endl;

    // Optionally print the matrices
    // std::cout << "Matrix A:" << std::endl;
    // print_matrix(A, rowsA, colsA);
    // std::cout << "Matrix B:" << std::endl;
    // print_matrix(B, rowsB, colsB);
    // std::cout << "Matrix C (Result):" << std::endl;
    // print_matrix(C, rowsA, colsB);

    return 0;
}
