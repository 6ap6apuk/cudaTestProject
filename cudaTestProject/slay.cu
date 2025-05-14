#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>

double EPS = 1.e-15;
int N = 10240;

__global__ void Solve(double* dA, double* dF, double* dX0, double* dX1, int size) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t < size) {
		double sum = 0.0;
		double aa = dA[t + t * size]; // ��� ����������������� �������
		for (int j = 0; j < size; j++) {
			// sum += dA[t + j * size] * dX0[j]; // ��� ������� �������
			sum += dA[j + t * size] * dX0[j];    // ��� ����������������� �������
		}
		dX1[t] = dX0[t] + (dF[t] - sum) / aa;
	}
}

__global__ void Eps(double* dX0, double* dX1, double* delta, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		delta[i] = fabs(dX1[i] - dX0[i]);
		dX0[i] = dX1[i]; // ���������� X0 ��� ��������� ��������
	}
}

int main() {
	float timerValueGPU, timerValueCPU;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	double* hA, * hF, * hX0, * hX, * hX1, * hDelta;
	double* dA, * dF, * dX0, * dX1, * delta;

	int size = N * N; // ������ ������� �������
	int N_thread = 512; // ����� ����� � �����
	int N_blocks = (N + N_thread - 1) / N_thread;
	unsigned int mem_sizeA = sizeof(double) * size; // ������ ��� �������
	unsigned int mem_sizeX = sizeof(double) * N; // ������ ��� ��������

	// ��������� ������ �� host
	hA = (double*)malloc(mem_sizeA); // ������� �
	hF = (double*)malloc(mem_sizeX); // ������ ����� ������� F
	hX = (double*)malloc(mem_sizeX); // ������ �������
	hX0 = (double*)malloc(mem_sizeX); // ������������ ������� X(n)
	hX1 = (double*)malloc(mem_sizeX); // ������������ ������� X(n+1)
	hDelta = (double*)malloc(mem_sizeX); // ������� |X(n+1)- X(n)|

	// ��������� ������ �� device
	cudaMalloc((void**)&dA, mem_sizeA); // ������� A
	cudaMalloc((void**)&dF, mem_sizeX); // ������ ����� F
	cudaMalloc((void**)&dX0, mem_sizeX); // ������� X(n)
	cudaMalloc((void**)&dX1, mem_sizeX); // ������� X(n+1)
	cudaMalloc((void**)&delta, mem_sizeX); // ������� |X(n+1)- X(n)|

	// ----------------------GPU ������� -------------------
	cudaMalloc((void**)&dA, mem_sizeA);
	cudaMalloc((void**)&dF, mem_sizeX);
	cudaMalloc((void**)&dX0, mem_sizeX);
	cudaMalloc((void**)&dX1, mem_sizeX);
	cudaMalloc((void**)&delta, mem_sizeX);

	// ����������� ������ c host �� device
	cudaMemcpy(dA, hA, mem_sizeA, cudaMemcpyHostToDevice); // ������� A
	cudaMemcpy(dF, hF, mem_sizeX, cudaMemcpyHostToDevice); // ������ ����� F
	cudaMemcpy(dX0, hX0, mem_sizeX, cudaMemcpyHostToDevice); // ���������
	// �����������

	// ���������� �������� ������
	for (int i = 0; i < N; i++) {
		hF[i] = 1.0;
		hX0[i] = 0.0;
		for (int j = 0; j < N; j++) {
			//hA[j + i * N] = (i == j) ? 2.0 : 0.1; // ������ ������� 
			hA[i + j * N] = (i == j) ? 2.0 : 0.1; // ������ �����������������
		}
	}

	// GPU Implementation
	{
		cudaEventRecord(start, 0);
		cudaMemcpy(dA, hA, mem_sizeA, cudaMemcpyHostToDevice);
		cudaMemcpy(dF, hF, mem_sizeX, cudaMemcpyHostToDevice);
		cudaMemcpy(dX0, hX0, mem_sizeX, cudaMemcpyHostToDevice);

		
		double eps = 1.0;
		int k = 0;
		while (eps > EPS) {
			k++;
			Solve << <N_blocks, N_thread >> > (dA, dF, dX0, dX1, N);
			Eps << <N_blocks, N_thread >> > (dX0, dX1, delta, N);

			cudaDeviceSynchronize(); // ������ �������������!

			cudaMemcpy(hDelta, delta, mem_sizeX, cudaMemcpyDeviceToHost);
			eps = 0.0;
			for (int j = 0; j < N; j++) eps += hDelta[j];
			eps /= N;
		}

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timerValueGPU, start, stop);
	}

	// CPU Implementation
	{
		auto start_cpu = std::chrono::high_resolution_clock::now();

		double* temp;
		double eps = 1.0;
		int k = 0;
		while (eps > EPS) {
			k++;
			eps = 0.0;

			for (int i = 0; i < N; i++) {
				double sum = 0.0;
				// ��� ������� �������: j + i*N
				// ��� �����������������: i + j*N
				for (int j = 0; j < N; j++)
					sum += hA[i + j * N] * hX0[j];

				hX1[i] = hX0[i] + (hF[i] - sum) / hA[i + i * N];
			}
			eps /= N;

			// ����� ���������� ������ ����������� ������
			temp = hX0;
			hX0 = hX1;
			hX1 = temp;
		}

		auto end_cpu = std::chrono::high_resolution_clock::now();
		timerValueCPU = std::chrono::duration<float, std::milli>(end_cpu - start_cpu).count();
	}
	printf("\n GPU calculation time: %f ms\n", timerValueGPU);
	printf("\n CPU calculation time: %f ms\n", timerValueCPU);
	// ����� ������������ ���������
	printf("\n Rate: %f x\n", timerValueCPU / timerValueGPU);

	// ������������ ������
	free(hA);
	free(hF);
	free(hX0);
	free(hX1);
	free(hDelta);
	cudaFree(dA);
	cudaFree(dF);
	cudaFree(dX0);
	cudaFree(dX1);
	cudaFree(delta);

	return 0;
}