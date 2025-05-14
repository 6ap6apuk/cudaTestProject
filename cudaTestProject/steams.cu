#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <chrono>

__global__ void function(float* dA, float* dB, float* dC, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	float ab, sum = 0.f;
	if (i < size)
	{
		ab = dA[i] * dB[i];
		for (j = 0; j < 100; j++) sum = sum + sinf(j + ab);
		dC[i] = sum;
	}
}

int main() {
    float timerValueGPU, timerValueCPU;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int nStream = 1; // число CUDA-потоков
    int size = 512 * 50000 / nStream; // размер каждого массива
    int N_thread = 512; // число нитей в блоке
    int N_blocks, i;
    // выделение памяти для массивов hA,hB,hC для host
    unsigned int mem_size = sizeof(float) * size;
    const int num_streams = 1;

    cudaStream_t streams[num_streams];

    float* hA, * hB, * hC;
    float* dA, * dB, * dC;

    // Выделение pinned-памяти
    cudaHostAlloc((void**)&hA, mem_size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hB, mem_size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hC, mem_size, cudaHostAllocDefault);

    // Инициализация данных
    for (i = 0; i < size; i++)
    {
        hA[i] = sinf(i); hB[i] = cosf(2.0f * i - 5.0f); hC[i] = 0.0f;
    }

    cudaMalloc((void**)&dA, mem_size);
    cudaMalloc((void**)&dB, mem_size);
    cudaMalloc((void**)&dC, mem_size);

    // Создание потоков
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
        
    cudaEventRecord(start, 0);
    // Асинхронные операции
    int chunk_size = size / num_streams;
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        cudaMemcpyAsync(dA + offset, hA + offset, chunk_size * sizeof(float),
            cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(dB + offset, hB + offset, chunk_size * sizeof(float),
            cudaMemcpyHostToDevice, streams[i]);
    }

    int threadsPerBlock = 512;
    int blocksPerGrid = (chunk_size + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        function <<< blocksPerGrid, threadsPerBlock, 0, streams[i] >>> (dA + offset, dB + offset, dC + offset, chunk_size);
    }

    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        cudaMemcpyAsync(hC + offset, dC + offset, chunk_size * sizeof(float),
            cudaMemcpyDeviceToHost, streams[i]);
    }
    // вычисления GPU варианта
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);
    printf("\n GPU calculation time: %f ms\n", timerValueGPU);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (i = 0; i < size; i++) {
        int j;
        float ab, sum = 0.f;
        ab = hA[i] * hB[i];
        for (j = 0; j < 100; j++) sum = sum + sinf(j + ab);
        hC[i] = sum;
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    timerValueCPU = std::chrono::duration<float, std::milli>(end_cpu - start_cpu).count();
    printf("\n CPU calculation time: %f ms\n", timerValueCPU);
    // Вывод коэффициента ускорения
    printf("\n Rate: %f x\n", timerValueCPU / timerValueGPU);

    // Синхронизация
    for (int i = 0; i < num_streams; i++)
        cudaStreamSynchronize(streams[i]);

    // Освобождение ресурсов
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}