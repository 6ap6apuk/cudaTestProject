#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <chrono>

#define BLOCK_SIZE 32
__global__ void kernel_global(double* a, double* b, int n, double* c)
{
	int bx = blockIdx.x; // ����� ����� �� x
	int by = blockIdx.y; // ����� ����� �� y
	int tx = threadIdx.x; // ����� ���� � ����� �� x
	int ty = threadIdx.y; // ����� ���� � ����� �� y
	float sum = 0.0f;
	int ia = n * (BLOCK_SIZE * by + ty); // ����� ������ �� A�
	int ib = BLOCK_SIZE * bx + tx; // ����� ������� �� B�
	int ic = ia + ib; // ����� �������� �� ђ
	// ���������� �������� ������� C
	for (int k = 0; k < n; k++) sum += a[ia + k] * b[ib + k * n];
	c[ic] = sum;
}

__global__ void kernel_smem(double* a, double* b, int n, double* c)
{
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	int aBegin = n * BLOCK_SIZE * by, aEnd = aBegin + n - 1;
	int bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
	float sum = 0.0f;
	__shared__ float as[BLOCK_SIZE][BLOCK_SIZE+1];
	__shared__ float bs[BLOCK_SIZE][BLOCK_SIZE+1];
	for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
	{
		// SMEM - 1
		as[tx][ty] = a[ia + n * ty + tx]; 
		bs[tx][ty] = b[ib + n * ty + tx];
		// SMEM - 3
		as[ty][tx] = a[ia + n * ty + tx];
		bs[ty][tx] = b[ib + n * ty + tx];
		//__syncthreads();
		for (int k = 0; k < BLOCK_SIZE; k++) 
			// SMEM - 1
			// sum += as[k][ty] * bs[tx][k];
			// SMEM - 3
			sum += as [ty][k] * bs [k][tx];
		//__syncthreads();
	}
	c[aBegin + bBegin + ty * n + tx] = sum;
}

int main()
{
	int N = 2048;
	int m, n, k;
	// �������� ����������-�������
	float timerValueGPU, timerValueCPU;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	int numBytes = N * N * sizeof(double);
	double* adev, * bdev, * cdev, * a, * b, * c, * cc, * bT;
	// ��������� ������ �� host
	a = (double*)malloc(numBytes); //������� A
	b = (double*)malloc(numBytes); //������� B
	bT = (double*)malloc(numBytes); //����������������� ������� B
	c = (double*)malloc(numBytes); //������� � ��� GPU-��������
	cc = (double*)malloc(numBytes); //������� � ��� CPU-��������
	// ������� ������� A, B � ����������������� ������� B
	for (n = 0; n < N; n++)
	{
		for (m = 0; m < N; m++)
		{
			a[m + n * N] = 2.0f * m + n; b[m + n * N] = m - n; bT[m + n * N] = n - m;
		}
	}
	// ������� ����� ����� � ������
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / threads.x, N / threads.y);
	// ��������� ������ �� GPU
	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&bdev, numBytes);
	cudaMalloc((void**)&cdev, numBytes);
	// ---------------- GPU-������� ------------------------
	// ����������� ������ A � B � host �� device
	cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);
	// ������ �������
	cudaEventRecord(start, 0);
	// ������ �������-����
	// !!!!!!! kernel_global <<< blocks, threads >> > (adev, bdev, N, cdev);
	kernel_smem << < blocks, threads >> > (adev, bdev, N, cdev);
	// ������ ������� ���������� GPU-��������
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("\n GPU calculation time %f msec\n", timerValueGPU);
	// �����������, ����������� ������� C � device �� host
	cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

	// -------------------- CPU-������� --------------------
	// ����� �������
	auto start_cpu = std::chrono::high_resolution_clock::now();
	// ���������� ������� C
	for (n = 0; n < N; n++)
	{
		for (m = 0; m < N; m++)
		{
			cc[m + n * N] = 0.f;
			for (k = 0; k < N; k++) cc[m + n * N] += a[k + n * N] * bT[k + m * N]; // bT !!!
		}
	}
	// ������ ������� ���������� CPU-��������
	auto end_cpu = std::chrono::high_resolution_clock::now();
	timerValueCPU = std::chrono::duration<float, std::milli>(end_cpu - start_cpu).count();
	printf("\n CPU calculation time: %f ms\n", timerValueCPU);
	printf("\n Rate %f x\n", timerValueCPU / timerValueGPU);
	// ������������ ������ �� GPU � CPU
	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);
	free(a);
	free(b);
	free(bT);
	free(c);
	free(cc);
	// ����������� ����������-�������
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}