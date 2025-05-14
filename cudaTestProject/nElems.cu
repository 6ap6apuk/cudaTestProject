#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <chrono>

// GPU-�������. ������ ���������
__global__ void Acceleration_GPU(float* X, float* Y,
	float* AX, float* AY, int nt, int N)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float ax = 0.f; float ay = 0.f; float xx, yy, rr; int sh = (nt - 1) * N;
	for (int j = 0; j < N; j++) // ���� �� ��������
	{
		if (j != id) // �������� ������������
		{
			xx = X[j + sh] - X[id + sh]; yy = Y[j + sh] - Y[id + sh];
			rr = sqrtf(xx * xx + yy * yy);
			if (rr > 0.01f) // ����������� ���������� 0.01 �
			{
				rr = 10.f / (rr * rr * rr); ax += xx * rr; ay += yy * rr;
			} // if rr
		} // if id
	} // for j
	AX[id] = ax; AY[id] = ay;
}

__global__ void Acceleration_Shared(float* X, float* Y, float* AX, float* AY,
	int nt, int N, int N_block)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float ax = 0.f; float ay = 0.f; float xx, yy, rr; int sh = (nt - 1) * N;
	float xxx = X[id + sh]; float yyy = Y[id + sh];
	__shared__ float Xs[256]; __shared__ float Ys[256]; // ��������� ����������� ������
	for (int i = 0; i < N_block; i++) // �������� ���� �� ������
	{
		Xs[threadIdx.x] = X[threadIdx.x + i * blockDim.x + sh]; // ����������� �� ����������
		Ys[threadIdx.x] = Y[threadIdx.x + i * blockDim.x + sh]; // � ����������� ������
		//__syncthreads(); // �������������
		for (int j = 0; j < blockDim.x; j++) // �������������� �����
		{
			if ((j + i * blockDim.x) != id)
			{
				xx = Xs[j] - xxx; yy = Ys[j] - yyy; rr = sqrtf(xx * xx + yy * yy);
				if (rr > 0.01f) { rr = 10.f / (rr * rr * rr); ax += xx * rr; ay += yy * rr; } //if
			} // if id
		} // for j
		//__syncthreads(); // �������������
	} // for i
	AX[id] = ax; AY[id] = ay;
}

// GPU-�������. �������� ���������
__global__ void Position_GPU(float* X, float* Y, float* VX, float* VY,
	float* AX, float* AY, float tau, int nt, int Np)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int sh = (nt - 1) * Np;
	X[id + nt * Np] = X[id + sh] + VX[id] * tau + AX[id] * tau * tau * 0.5f;
	Y[id + nt * Np] = Y[id + sh] + VY[id] * tau + AY[id] * tau * tau * 0.5f;
	VX[id] += AX[id] * tau;
	VY[id] += AY[id] * tau;
}

// CPU � �������. ���������� ���������
void Acceleration_CPU(float* X, float* Y, float* AX, float* AY,
	int nt, int N, int id)
{
	float ax = 0.f; float ay = 0.f; float xx, yy, rr; int sh = (nt - 1) * N;
	for (int j = 0; j < N; j++) // ���� �� ��������
	{
		if (j != id) // ������� ������������
		{
			xx = X[j + sh] - X[id + sh]; yy = Y[j + sh] - Y[id + sh];
			rr = sqrtf(xx * xx + yy * yy);
			if (rr > 0.01f) // ����������� ���������� 0.01 �
			{
				rr = 10.f / (rr * rr * rr); ax += xx * rr; ay += yy * rr;
			} // if rr
		} // if id
	} // for
	AX[id] = ax; AY[id] = ay;
}

// CPU-�������. �������� ���������
void Position_CPU(float* X, float* Y, float* VX,
	float* VY, float* AX, float* AY,
	float tau, int nt, int Np, int id)
{
	int sh = (nt - 1) * Np;
	X[id + nt * Np] = X[id + sh] + VX[id] * tau + AX[id] * tau * tau * 0.5f;
	Y[id + nt * Np] = Y[id + sh] + VY[id] * tau + AY[id] * tau * tau * 0.5f;
	VX[id] += AX[id] * tau;
	VY[id] += AY[id] * tau;
}


int main(int argc, char* argv[]) {
	float timerValueGPU, timerValueCPU;
	cudaEvent_t start, stop;
		// ������������� ������� CUDA
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int N = 10240; // ����� ������ (2-� ������� 20480)
	int NT = 10; // ����� ����� �� ������� (��� �������� - 800)
	float tau = 0.001f; // ��� �� ������� 0.001 �
	// �������� �������� �� host
	float* hX, * hY, * hVX, * hVY, * hAX, * hAY;
	unsigned int mem_size = sizeof(float) * N;
	unsigned int mem_size_big = sizeof(float) * NT * N;
	hX = (float*)malloc(mem_size_big); hY = (float*)malloc(mem_size_big);
	hVX = (float*)malloc(mem_size); hVY = (float*)malloc(mem_size);
	hAX = (float*)malloc(mem_size); hAY = (float*)malloc(mem_size);
	// ������� ��������� ������� �� host
	float vv, phi;
	for (int j = 0; j < N; j++)
	{
		phi = (float)rand();
		hX[j] = rand() * cosf(phi) * 1.e-4f; hY[j] = rand() * sinf(phi) * 1.e-4f;
		vv = (hX[j] * hX[j] + hY[j] * hY[j]) * 10.f;
		hVX[j] = -vv * sinf(phi); hVY[j] = vv * cosf(phi);
	}
	// �������� �� device ��������
	float* dX, * dY, * dVX, * dVY, * dAX, * dAY;
	cudaMalloc((void**)&dX, mem_size_big);
	cudaMalloc((void**)&dY, mem_size_big);
	cudaMalloc((void**)&dVX, mem_size); cudaMalloc((void**)&dVY, mem_size);
	cudaMalloc((void**)&dAX, mem_size); cudaMalloc((void**)&dAY, mem_size);
	// ������� ����� ����� � ������
	int N_thread = 256; int N_block = N / N_thread;

	// -----------------GPU-�������--------------------------
	cudaEventRecord(start, 0);
	// ����������� ������ �� device
	cudaMemcpy(dX, hX, mem_size_big, cudaMemcpyHostToDevice);
	cudaMemcpy(dY, hY, mem_size_big, cudaMemcpyHostToDevice);
	cudaMemcpy(dVX, hVX, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dVY, hVY, mem_size, cudaMemcpyHostToDevice);

	for (int j = 1; j < NT; j++) {
		Acceleration_Shared << <N_block, N_thread >> > (dX, dY, dAX, dAY, j, N, N_block);
		Position_GPU << <N_block, N_thread >> > (dX, dY, dVX, dVY, dAX, dAY, tau, j, N);

		// �������� ������ ����� ������� ����
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
			exit(1);
		}
	}

	// ������������� ����� ������� �������
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("\n GPU calculation time %f msec\n", timerValueGPU);

	auto cpu_start = std::chrono::high_resolution_clock::now(); // ����������

	for (int j = 1; j < NT; j++) {
		for (int id = 0; id < N; id++) {
			Acceleration_CPU(hX, hY, hAX, hAY, j, N, id);
			Position_CPU(hX, hY, hVX, hVY, hAX, hAY, tau, j, N, id);
		}
	}

	auto cpu_end = std::chrono::high_resolution_clock::now();
	timerValueCPU = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count(); // ����������

	printf("\n CPU calculation time %f msec\n", timerValueCPU);
	printf("\n Rate %f x\n", timerValueCPU / timerValueGPU);

	// ������������ ������
	free(hX); free(hY); free(hVX); free(hVY); free(hAX); free(hAY);
	cudaFree(dX); cudaFree(dY); cudaFree(dVX); cudaFree(dVY);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}