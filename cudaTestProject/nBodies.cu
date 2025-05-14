#define _USE_MATH_DEFINES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <chrono>

#include <curand_kernel.h>  // Для генерации случайных чисел на GPU

// Параметры симуляции
#define NUM_PARTICLES 20480    // Количество частиц
#define TIME_STEPS 10          // Число временных шагов
#define BLOCK_SIZE 256         // Оптимальный размер блока
#define MIN_DISTANCE 0.01f     // Минимальное расстояние взаимодействия
#define FORCE_FACTOR 10.0f     // Множитель силы
#define TAU 0.001f             // Шаг по времени

#define N 20480
#define BLOCK_SIZE 256
#define NT 10

__global__ void nbodyKernel(
    float* X, float* Y, float* VX, float* VY, float tau, int nt
) {

    extern __shared__ float s_data[];

    float* Xs = s_data;
    float* Ys = &s_data[blockDim.x];

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int sh = (nt - 1) * N;

    float ax = 0.0f, ay = 0.0f;

    // Загрузка данных в разделяемую память
    Xs[threadIdx.x] = X[id + sh];
    Ys[threadIdx.x] = Y[id + sh];
    __syncthreads();

    for (int j = 0; j < blockDim.x; j++) {
        if (j + blockIdx.x * blockDim.x != id) {
            float xx = Xs[j] - X[id + sh];
            float yy = Ys[j] - Y[id + sh];
            float rr = sqrtf(xx * xx + yy * yy);
            if (rr > 0.01f) {
                rr = 10.0f / (rr * rr * rr);
                ax += xx * rr;
                ay += yy * rr;
            }
        }
    }

    // Обновление позиций и скоростей
    X[id + nt * N] = X[id + sh] + VX[id] * tau + ax * tau * tau * 0.5f;
    Y[id + nt * N] = Y[id + sh] + VY[id] * tau + ay * tau * tau * 0.5f;
    VX[id] += ax * tau;
    VY[id] += ay * tau;
}

int main() {
    float* hX, * hY, * hVX, * hVY;
    float* dX, * dY, * dVX, * dVY;

    size_t mem_size = N * sizeof(float);
    size_t mem_size_big = NT * N * sizeof(float);

    float tau = 0.001f;

    // Выделение памяти на хосте
    hX = (float*)malloc(mem_size_big);
    hY = (float*)malloc(mem_size_big);
    hVX = (float*)malloc(mem_size);
    hVY = (float*)malloc(mem_size);

    // Инициализация данных
    for (int i = 0; i < N; i++) {
        float phi = (float)rand() / RAND_MAX * 2 * M_PI;
        hX[i] = cosf(phi) * 1e-4f;
        hY[i] = sinf(phi) * 1e-4f;
        float vv = (hX[i] * hX[i] + hY[i] * hY[i]) * 10.0f;
        hVX[i] = -vv * sinf(phi);
        hVY[i] = vv * cosf(phi);
    }

    // Выделение памяти на устройстве
    cudaMalloc((void**)&dX, mem_size_big);
    cudaMalloc((void**)&dY, mem_size_big);
    cudaMalloc((void**)&dVX, mem_size);
    cudaMalloc((void**)&dVY, mem_size);
    // Копирование данных на устройство
    cudaMemcpy(dX, hX, mem_size_big, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, hY, mem_size_big, cudaMemcpyHostToDevice);
    cudaMemcpy(dVX, hVX, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dVY, hVY, mem_size, cudaMemcpyHostToDevice);
    // Настройка параметров запуска
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Таймеры
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpuTime;

    // Запуск GPU-расчета
    cudaEventRecord(start);
    for (int j = 1; j < NT; j++) {
        nbodyKernel << <blocks, threads, 2 * BLOCK_SIZE * sizeof(float) >> > (dX, dY, dVX, dVY, tau, j);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("GPU calculation time: %.2f ms\n", gpuTime);

    // Освобождение памяти
    free(hX); free(hY); free(hVX); free(hVY);
    cudaFree(dX); cudaFree(dY); cudaFree(dVX); cudaFree(dVY);

    return 0;
}