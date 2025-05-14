#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <chrono>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void function(float* dA, float* dB, float* dC, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) dC[i] = dA[i] + dB[i];
}

int main(int argc, char* argv[])
{// инициализация переменных-событий для таймера
    float timerValueGPU, timerValueCPU;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float* hA, * hB, * hC, * dA, * dB, * dC;
    int size = 512 * 50000; // размер каждого массива
    int N_thread = 512; // число нитей в блоке
    int N_blocks, i;
    // задание массивов hA, hB, hC для host
    unsigned int mem_size = sizeof(float) * size;
    cudaHostAlloc((void**)&hA, mem_size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hB, mem_size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hC, mem_size, cudaHostAllocDefault);
    // выделение памяти на device под массивы hA, hB, hC
    cudaMalloc((void**)&dA, mem_size);
    cudaMalloc((void**)&dB, mem_size);
    cudaMalloc((void**)&dC, mem_size);
    // заполнение массивов hA,hB и обнуление hC
    for (i = 0; i < size; i++)
    {
        hA[i] = 1.0f / ((i + 1.0f) * (i + 1.0f));
        hB[i] = expf(1.0f / (i + 1.0f));
        hC[i] = 0.0f;
    }
    // определение числа блоков
    if ((size % N_thread) == 0)
    {
        N_blocks = size / N_thread;
    }
    else
    {
        N_blocks = (int)(size / N_thread) + 1;
    }
    dim3 blocks(N_blocks);

    // ----------------------GPU вариант -------------------
    // Старт таймера
    //cudaEventRecord(start, 0);
    // Копирование массивов с host на device
    cudaMemcpy(dA, hA, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, mem_size, cudaMemcpyHostToDevice);
    // Запуск функции-ядра
    cudaEventRecord(start, 0);
    function <<< N_blocks, N_thread >>> (dA, dB, dC, size);
    // Копирование результат с device на host
    cudaMemcpy(hC, dC, mem_size, cudaMemcpyDeviceToHost);
    // Остановка таймера и вывод времени
    // вычисления GPU варианта
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);
    printf("\n GPU calculation time: %f ms\n", timerValueGPU);

    // --------------------- CPU вариант --------------------
    // Старт таймера
    auto start_cpu = std::chrono::high_resolution_clock::now();
    // cudaEventRecord(start, 0);
    // вычисления
    for (i = 0; i < size; i++) hC[i] = hA[i] + hB[i];
    // Остановка таймера и вывод времени
    // вычисления СPU варианта
    //cudaEventRecord(stop, 0);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&timerValueCPU, start, stop);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    timerValueCPU = std::chrono::duration<float, std::milli>(end_cpu - start_cpu).count();
    printf("\n CPU calculation time: %f ms\n", timerValueCPU);
    // Вывод коэффициента ускорения
    printf("\n Rate: %f x\n", timerValueCPU / timerValueGPU);
    // Освобождение памяти на host и device
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    // уничтожение переменных-событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
