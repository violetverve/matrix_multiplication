#include "rknn_api.h"
#include "rknn_matmul_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Utility function to allocate and initialize a tensor memory
rknn_tensor_mem* allocate_tensor_mem(rknn_context ctx, rknn_matmul_tensor_attr* attr, void* data, uint32_t size) {
    rknn_tensor_mem* mem = rknn_create_mem(ctx, size);
    if (mem == NULL) {
        printf("Failed to allocate tensor memory.\n");
        return NULL;
    }
    memcpy(mem->virt_addr, data, size);
    return mem;
}

int main() {
    rknn_matmul_ctx ctx;

    // Matrix dimensions
    const int M = 2016;
    const int K = 2016; // Ensure K is aligned with 32 (for RKNN requirements)
    const int N = 2016; // Ensure N is aligned with 16 (for RKNN requirements)

    // Matmul info
    rknn_matmul_info info;
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32; // Adjust based on your requirement
    info.B_layout = 0;
    info.AC_layout = 0;

    // Tensor attributes for matrices A, B, and C
    rknn_matmul_tensor_attr A_attr;
    memset(&A_attr, 0, sizeof(rknn_matmul_tensor_attr));
    strcpy(A_attr.name, "A");
    A_attr.n_dims = 2;
    A_attr.dims[0] = M;
    A_attr.dims[1] = K;
    A_attr.size = M * K * sizeof(float);
    A_attr.type = RKNN_TENSOR_FLOAT32;

    rknn_matmul_tensor_attr B_attr;
    memset(&B_attr, 0, sizeof(rknn_matmul_tensor_attr));
    strcpy(B_attr.name, "B");
    B_attr.n_dims = 2;
    B_attr.dims[0] = K;
    B_attr.dims[1] = N;
    B_attr.size = K * N * sizeof(float);
    B_attr.type = RKNN_TENSOR_FLOAT32;

    rknn_matmul_tensor_attr C_attr;
    memset(&C_attr, 0, sizeof(rknn_matmul_tensor_attr));
    strcpy(C_attr.name, "C");
    C_attr.n_dims = 2;
    C_attr.dims[0] = M;
    C_attr.dims[1] = N;
    C_attr.size = M * N * sizeof(float);
    C_attr.type = RKNN_TENSOR_FLOAT32;

    // Input/output attributes
    rknn_matmul_io_attr io_attr;
    memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));
    io_attr.A = A_attr;
    io_attr.B = B_attr;
    io_attr.C = C_attr;

    // Allocate memory for matrices A, B, and C
    float* A_data = (float*)malloc(A_attr.size);
    float* B_data = (float*)malloc(B_attr.size);

    if (A_data == NULL || B_data == NULL) {
        printf("Failed to allocate memory for matrix data.\n");
        return -1;
    }

       // Initialize matrices A and B with float values including decimals
    for (int i = 0; i < M * K; ++i) {
        A_data[i] = static_cast<float>(i + 1) / 100.0f; // Example initialization with decimals
    }

    for (int i = 0; i < K * N; ++i) {
        B_data[i] = static_cast<float>(i + 1) / 100.0f; // Example initialization with decimals
    }

    // Create matmul context
    if (rknn_matmul_create(&ctx, &info, &io_attr) != RKNN_SUCC) {
        printf("Failed to create matmul context.\n");
        free(A_data);
        free(B_data);
        return -1;
    }

    // Allocate tensor memory for matrices A, B, and C
    rknn_tensor_mem* A_mem = allocate_tensor_mem(ctx, &A_attr, A_data, A_attr.size);
    rknn_tensor_mem* B_mem = allocate_tensor_mem(ctx, &B_attr, B_data, B_attr.size);
    rknn_tensor_mem* C_mem = rknn_create_mem(ctx, C_attr.size);

    if (A_mem == NULL || B_mem == NULL || C_mem == NULL) {
        printf("Failed to allocate tensor memory.\n");
        free(A_data);
        free(B_data);
        return -1;
    }

    // Set input/output memory
    if (rknn_matmul_set_io_mem(ctx, A_mem, &A_attr) != RKNN_SUCC ||
        rknn_matmul_set_io_mem(ctx, B_mem, &B_attr) != RKNN_SUCC ||
        rknn_matmul_set_io_mem(ctx, C_mem, &C_attr) != RKNN_SUCC) {
        printf("Failed to set input/output memory.\n");
        free(A_data);
        free(B_data);
        return -1;
    }

    // Run matrix multiplication
    if (rknn_matmul_run(ctx) != RKNN_SUCC) {
        printf("Failed to run matmul.\n");
        free(A_data);
        free(B_data);
        return -1;
    }

    // // Print the result matrix C
    // float* C_data = (float*)C_mem->virt_addr;
    // printf("Result matrix C:\n");a
    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         printf("%f ", C_data[i * N + j]);
    //     }
    //     printf("\n");
    // }

    // Clean up allocated resources
    free(A_data);
    free(B_data);
    rknn_destroy_mem(ctx, A_mem);
    rknn_destroy_mem(ctx, B_mem);
    rknn_destroy_mem(ctx, C_mem);
    rknn_destroy(ctx);

    return 0;
}
