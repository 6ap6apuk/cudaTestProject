#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <chrono>

#define D(i, j) ((i) == (j) ? 1.0 : 0.0) // Дельта-функция Кронекера

__global__ void Matrix_A(double* dA, double* dX, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) { // Добавлена проверка границ
        int Nd = (int)(0.15 * N);
        if (i <= j + Nd && i >= j - Nd) {
            dA[i + j * N] = pow(sin(dX[j]) * cos(dX[i]), 2.) + (double)N * D(i, j);
        }
        else {
            dA[i + j * N] = 0.;
        }
    }
}

__global__ void AX(double* dAX, double* dA, double* dX, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { // Добавлена проверка границ
        double sum = 0.;
        for (int j = 0; j < N; j++)
            sum += dA[i + j * N] * dX[j];
        dAX[i] = sum;
    }
}

// Реализация недостающих ядер
__global__ void Solve_L(double* dL, double* dPhi, double* dV0, double* dV1, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            if (j != i) sum += dL[i * N + j] * dV0[j];
        }
        dV1[i] = (dPhi[i] - sum) / dL[i * N + i];
    }
}

__global__ void Eps_L(double* dV0, double* dV1, double* d_dV, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_dV[i] = fabs(dV1[i] - dV0[i]);
        dV0[i] = dV1[i];
    }
}


__global__ void PHI(double* dPhi, double* dAX, double* dF)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    dPhi[i] = dAX[i] - dF[i];
}

__global__ void D_PHI(double* dL, double* dX0, int N)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    double sum1, sum2 = 0.; int k, k1, k2, Nd = (int)(0.15 * N);
    if (i <= j + Nd && i >= j - Nd)
    {
        if (i >= 0 && i <= Nd) { k1 = 0; k2 = i + Nd + 1; } // область 1
        if (i >= Nd + 1 && i < N - Nd) { k1 = i - Nd; k2 = i + Nd + 1; } // область 2
        if (i >= N - Nd && i < N) { k1 = i - Nd; k2 = N; } // область 3
        for (k = k1; k < k2; k++) {
            sum1 += D(k, j) * (pow(sin(dX0[i]) * cos(dX0[k]), 2.) + D(i, k) * (double)N);
            sum2 += dX0[k] * (sin(2. * dX0[i]) * pow(cos(dX0[k]), 2.) * D(i, j) -
                sin(2. * dX0[k]) * pow(sin(dX0[i]), 2.) * D(k, j));
        } // k
        dL[i + j * N] = sum1 + sum2; // dLT !
    }
    else { dL[i + j * N] = 0.; }
}

__global__ void Solve_G(double* dX0, double* dX1, double* dV0,
    double tau)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    dX1[i] = dX0[i] + tau * dV0[i];
}
__global__ void Eps_G(double* dX0, double* dX1, double* d_dX)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_dX[i] = abs((long)dX0[i] - (long)dX1[i]);
    dX0[i] = dX1[i];
}

void CPU_PHI(double* phi, double* AX, double* F, int N) {
    for (int i = 0; i < N; i++) {
        phi[i] = AX[i] - F[i];
    }
}

void CPU_Solve_L(double* L, double* Phi, double* V0, double* V1, int N) {
    // Простейшая реализация метода Гаусса-Зейделя
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            if (j != i) sum += L[i * N + j] * V0[j];
        }
        V1[i] = (Phi[i] - sum) / L[i * N + i];
    }
}

void CPU_Eps_L(double* V0, double* V1, double* dV, int N) {
    for (int i = 0; i < N; i++) {
        dV[i] = fabs(V1[i] - V0[i]);
        V0[i] = V1[i];
    }
}

void CPU_Solve_G(double* X0, double* X1, double* V0, double tau, int N) {
    for (int i = 0; i < N; i++) {
        X1[i] = X0[i] + tau * V0[i];
    }
}

void CPU_Eps_G(double* X0, double* X1, double* dX, int N) {
    for (int i = 0; i < N; i++) {
        dX[i] = fabs(X1[i] - X0[i]);
        X0[i] = X1[i];
    }
}

// CPU аналоги функций
void CPU_Matrix_A(double* A, double* X, int N) {
    int Nd = (int)(0.15 * N);
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            if (i <= j + Nd && i >= j - Nd) {
                A[i + j * N] = pow(sin(X[j]) * cos(X[i]), 2.) + N * D(i, j);
            }
            else {
                A[i + j * N] = 0.0;
            }
        }
    }
}

void CPU_AX(double* AX, double* A, double* X, int N) {
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i + j * N] * X[j];
        }
        AX[i] = sum;
    }
}

const int N = 2048;
const double EPS_G = 1e-6;
const double EPS_L = 1e-15;
const double tau = 0.1;

int main() {
    const int N = 2048;
    const double EPS_G = 1e-6;
    const double EPS_L = 1e-15;
    const double tau = 0.1;

    // Выделение памяти GPU
    double* dA, * dX0, * dX1, * dF, * dPhi, * dL, * dV0, * dV1, * dAX, * d_dV, * d_dX;
    double* hA, * hX1, * hP, * hL, * hV0, * hV1, * h_dV;
    cudaMalloc(&dA, N * N * sizeof(double));
    cudaMalloc(&dAX, N * sizeof(double));
    cudaMalloc(&dX0, N * sizeof(double));
    cudaMalloc(&dX1, N * sizeof(double));
    cudaMalloc(&dF, N * sizeof(double));
    cudaMalloc(&dPhi, N * sizeof(double));
    cudaMalloc(&dL, N * N * sizeof(double));
    cudaMalloc(&dV0, N * sizeof(double));
    cudaMalloc(&dV1, N * sizeof(double));
    cudaMalloc(&d_dV, N * sizeof(double));
    cudaMalloc(&d_dX, N * sizeof(double));

    // Инициализация данных
    double* hF = (double*)malloc(N * sizeof(double));
    double* hX0 = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        hX0[i] = 0.1;
        hF[i] = 1.0;
    }
    cudaMemcpy(dX0, hX0, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dF, hF, N * sizeof(double), cudaMemcpyHostToDevice);

    // Настройка параметров запуска
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // GPU вычисления
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    double eps_G = 1.0;
    int iter_G = 0;
    while (eps_G > EPS_G && iter_G < 1000) {
        iter_G++;

        // Шаг 1: Вычисление матрицы A
        Matrix_A << < grid, block >> > (dA, dX0, N);

        // Шаг 2: Вычисление A*x
        AX << < (N + 255) / 256, 256 >> > (dAX, dA, dX0, N);

        // Шаг 3: Вычисление невязки
        PHI << < (N + 255) / 256, 256 >> > (dPhi, dAX, dF);

        // Шаг 4: Вычисление матрицы Якоби
        D_PHI << < grid, block >> > (dL, dX0, N);

        // Решение СЛАУ
        double eps_L = 1.0;
        int iter_L = 0;
        cudaMemset(dV0, 0, N * sizeof(double));
        while (eps_L > EPS_L && iter_L < 1000) {
            Solve_L << < (N + 255) / 256, 256 >> > (dL, dPhi, dV0, dV1, N);
            Eps_L << < (N + 255) / 256, 256 >> > (dV0, dV1, d_dV, N);

            // Копирование и расчет погрешности
            double h_dV[N];
            cudaMemcpy(h_dV, d_dV, N * sizeof(double), cudaMemcpyDeviceToHost);
            eps_L = 0.0;
            for (int j = 0; j < N; j++) eps_L += h_dV[j];
            eps_L /= N;
            iter_L++;
        }

        // Обновление решения
        Solve_G << < (N + 255) / 256, 256 >> > (dX0, dX1, dV0, tau);
        Eps_G << < (N + 255) / 256, 256 >> > (dX0, dX1, d_dX);

        // Расчет погрешности
        double h_dX[N];
        cudaMemcpy(h_dX, d_dX, N * sizeof(double), cudaMemcpyDeviceToHost);
        eps_G = 0.0;
        for (int k = 0; k < N; k++) eps_G += h_dX[k];
        eps_G /= N;

        std::swap(dX0, dX1);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // CPU вычисления
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // Выделение памяти для CPU вычислений
    double* hA_cpu = (double*)malloc(N * N * sizeof(double));
    double* hAX_cpu = (double*)malloc(N * sizeof(double));
    double* hPhi_cpu = (double*)malloc(N * sizeof(double));
    double* hL_cpu = (double*)malloc(N * N * sizeof(double));
    double* hX0_cpu = (double*)malloc(N * sizeof(double));
    double* hX1_cpu = (double*)malloc(N * sizeof(double));
    double* hV0_cpu = (double*)malloc(N * sizeof(double));
    double* hV1_cpu = (double*)malloc(N * sizeof(double));
    double* h_dV_cpu = (double*)malloc(N * sizeof(double));
    double* h_dX_cpu = (double*)malloc(N * sizeof(double));

    // Инициализация начальных данных
    memcpy(hX0_cpu, hX0, N * sizeof(double));
    for (int i = 0; i < N; i++) hF[i] = 1.0;

    double eps_G_cpu = 1.0;
    int iter_G_cpu = 0;

    // Реализация функции CPU_D_PHI
    auto CPU_D_PHI = [](double* L, double* X0, int N) {
        int Nd = (int)(0.15 * N);
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                if (i <= j + Nd && i >= j - Nd) {
                    double sum1 = 0.0, sum2 = 0.0;
                    int k1, k2;

                    if (i >= 0 && i <= Nd) { k1 = 0; k2 = i + Nd + 1; }
                    else if (i >= Nd + 1 && i < N - Nd) { k1 = i - Nd; k2 = i + Nd + 1; }
                    else { k1 = i - Nd; k2 = N; }

                    for (int k = k1; k < k2; k++) {
                        sum1 += D(k, j) * (pow(sin(X0[i]) * cos(X0[k]), 2.) + D(i, k) * (double)N);
                        sum2 += X0[k] * (sin(2. * X0[i]) * pow(cos(X0[k]), 2.) * D(i, j)
                            - sin(2. * X0[k]) * pow(sin(X0[i]), 2.) * D(k, j));
                    }
                    L[i + j * N] = sum1 + sum2;
                }
                else {
                    L[i + j * N] = 0.0;
                }
            }
        }
        };

    while (eps_G_cpu > EPS_G && iter_G_cpu < 1000) {
        iter_G_cpu++;

        // 1. Вычисление матрицы A(x_n)
        CPU_Matrix_A(hA_cpu, hX0_cpu, N);

        // 2. Вычисление A(x_n)x_n
        CPU_AX(hAX_cpu, hA_cpu, hX0_cpu, N);

        // 3. Вычисление невязки Φ(x_n) = A(x_n)x_n - f
        CPU_PHI(hPhi_cpu, hAX_cpu, hF, N);

        // 4. Вычисление матрицы Якоби L(x_n)
        CPU_D_PHI(hL_cpu, hX0_cpu, N);

        // 5. Решение линейной системы L(x_n)v_n = Φ(x_n)
        double eps_L_cpu = 1.0;
        int iter_L_cpu = 0;
        memset(hV0_cpu, 0, N * sizeof(double));

        while (eps_L_cpu > EPS_L && iter_L_cpu < 1000) {
            iter_L_cpu++;
            CPU_Solve_L(hL_cpu, hPhi_cpu, hV0_cpu, hV1_cpu, N);
            CPU_Eps_L(hV0_cpu, hV1_cpu, h_dV_cpu, N);

            eps_L_cpu = 0.0;
            for (int j = 0; j < N; j++) eps_L_cpu += h_dV_cpu[j];
            eps_L_cpu /= N;
        }

        // 6. Обновление решения x_{n+1} = x_n + τv_n
        CPU_Solve_G(hX0_cpu, hX1_cpu, hV0_cpu, tau, N);

        // 7. Расчет новой погрешности
        CPU_Eps_G(hX0_cpu, hX1_cpu, h_dX_cpu, N);

        eps_G_cpu = 0.0;
        for (int k = 0; k < N; k++) eps_G_cpu += h_dX_cpu[k];
        eps_G_cpu /= N;

        // 8. Обмен указателей для следующей итерации
        std::swap(hX0_cpu, hX1_cpu);
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

    // Освобождение CPU памяти
    free(hA_cpu);
    free(hAX_cpu);
    free(hPhi_cpu);
    free(hL_cpu);
    free(hX0_cpu);
    free(hX1_cpu);
    free(hV0_cpu);
    free(hV1_cpu);
    free(h_dV_cpu);
    free(h_dX_cpu);

    return 0;
}