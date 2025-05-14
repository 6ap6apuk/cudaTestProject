#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <chrono>

#define N 2048
#define BLOCK_SIZE 32
#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    }}

__global__ void matrixMulOptimized(float* A, float* B, float* C) {
    // Выделение разделяемой памяти для тайлов
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    for (int m = 0; m < N / BLOCK_SIZE; ++m) {
        // Загрузка тайлов в разделяемую память
        s_A[ty][tx] = A[row * N + (m * BLOCK_SIZE + tx)];
        s_B[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        //__syncthreads();

        // Вычисление произведения для загруженных тайлов
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        //__syncthreads();
    }

    // Запись результата
    if (row < N && col < N)
        C[row * N + col] = sum;
}

void cpuMatrixMul(float* A, float* B, float* C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    float* h_A, * h_B, * h_C, * h_C_ref;
    float* d_A, * d_B, * d_C;

    size_t size = N * N * sizeof(float);

    // Выделение и инициализация хостовой памяти
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    h_C_ref = (float*)malloc(size);

    // Генерация случайных данных
    srand(time(NULL));
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Выделение памяти на устройстве
    CHECK(cudaMalloc(&d_A, size));
    CHECK(cudaMalloc(&d_B, size));
    CHECK(cudaMalloc(&d_C, size));

    // Копирование данных на устройство
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Настройка параметров запуска
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + threads.x - 1) / threads.x,
        (N + threads.y - 1) / threads.y);

    // Измерение времени выполнения GPU
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    matrixMulOptimized <<<blocks, threads >> > (d_A, d_B, d_C);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float gpuTime;
    CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    // Копирование результатов обратно
    CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Валидация результатов
    printf("Validating results...\n");
    cpuMatrixMul(h_A, h_B, h_C_ref);

    float maxError = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        maxError = fmax(maxError, fabs(h_C_ref[i] - h_C[i]));
    }
    printf("Max error: %f\n", maxError);

    // Освобождение памяти
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    printf("GPU Time: %.2f ms\n", gpuTime);
    return 0;
}