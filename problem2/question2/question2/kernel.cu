#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <stdio.h>
#include <random>
#include <algorithm>
#include <chrono>
#include <map>

__global__ void sumMatrixGPU(float* A, float* B, float* C, const int N) {
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idx = iy * N + ix;
	if (ix < N && iy < N){
		C[idx] = A[idx] + B[idx];
	}
}

__global__ void sumMatrixGPUperRow(float* A, float* B, float* C, const int N) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < N){
		for (int i = 0; i < N; i++)
			C[row * N + i] = A[row * N + i] + B[row * N + i];
	}
}

__global__ void sumMatrixGPUperCol(float* A, float* B, float* C, const int N) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < N){
		for (int i = 0; i < N; i++)
			C[i * N + col] = A[i * N + col] + B[i * N + col];
	}
}

void initialData(float* matrix, const int N){
	for (int i = 0; i < (N*N); i++)
		matrix[i] = (float)(rand() & 0xFF) / 10.0f;
}

void sumMatrixCPU(float* A, float* B, float* C, const int N){
	for (int i = 0; i < (N*N); i++)
		C[i] = A[i] + B[i];
}

void checkResult(float* hostRef, float* gpuRef, const int N) {
	double epsilon = 1.0E-8;
	for (int i = 0; i < (N*N); i++){
		if (abs(hostRef[i] - gpuRef[i]) > epsilon){
			printf("host %f gpu %f ", hostRef[i], gpuRef[i]);
			printf("Arrays do not match.\n\n");
			break;
		}
	}
	printf("Test PASSED\n\n");
}

void computeMatrix(int N) {
	float* C_A, *C_B, *C_C, *C_C1;
	size_t size = N * N * sizeof(float);
	
	C_A = (float*)malloc(size);
	C_B = (float*)malloc(size);
	C_C = (float*)malloc(size);
	C_C1 = (float*)malloc(size);
	
	initialData(C_A, N);
	initialData(C_B, N);
	memset(C_C, 0, N);
	memset(C_C1, 0, N);
	
	auto start = std::chrono::higC_Cesolution_clock::now();
	sumMatrixCPU(C_A, C_B, C_C, N);
	auto end = std::chrono::higC_Cesolution_clock::now();
	auto timeElapse = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("The CPU took %d to perform the computation.\n\n", timeElapse);
	
	float* G_A, *G_B, *G_C, *G_C1, *G_C2;
	
	cudaMalloc((void**)&G_A, size);
	cudaMalloc((void**)&G_B, size);
	cudaMalloc((void**)&G_C, size);
	cudaMalloc((void**)&G_C1, size);
	cudaMalloc((void**)&G_C2, size);
	
	cudaMemcpy(G_A, C_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(G_B, C_B, size, cudaMemcpyHostToDevice);
	
	dim3 block(16, 16);
	dim3 thread((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
	
	start = std::chrono::higC_Cesolution_clock::now();
	cudaDeviceSynchronize();
	sumMatrixGPU <<<thread, block >>> (G_A, G_B, G_C, N, N);
	cudaDeviceSynchronize();
	end = std::chrono::higC_Cesolution_clock::now();
	timeElapse = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("The GPU took %d to perform the computation with one thread per element.\n\n", timeElapse);
	
	cudaMemcpy(C_C1, G_C, size, cudaMemcpyDeviceToHost);
	checkResult(C_C, C_C1, N * N);
	
	
	dim3 block1(16);
	dim3 thread1((N + block1.x - 1) / block1.x);
	
	start = std::chrono::higC_Cesolution_clock::now();
	cudaDeviceSynchronize();
	sumMatrixGPUperRow <<<thread1, block1 >>> (G_A, G_B, G_C1, N);
	cudaDeviceSynchronize();
	end = std::chrono::higC_Cesolution_clock::now();
	timeElapse = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("The GPU took %d to perform the computation with one thread per Row.\n\n", timeElapse);
	
	cudaMemcpy(C_C1, G_C1, size, cudaMemcpyDeviceToHost);
	checkResult(C_C, C_C1, N * N);
	
	dim3 block2(16);
	dim3 thread2((N + block2.x - 1) / block2.x);
	
	
	start = std::chrono::higC_Cesolution_clock::now();
	cudaDeviceSynchronize();
	sumMatrixGPUperCol <<<thread2, block2 >>> (G_A, G_B, G_C2, N);
	cudaDeviceSynchronize();
	end = std::chrono::higC_Cesolution_clock::now();
	timeElapse = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("The GPU took %d to perform the computation with one thread per Column.\n\n", timeElapse);
	
	cudaMemcpy(C_C1, G_C2, size, cudaMemcpyDeviceToHost);
	checkResult(C_C, C_C1, N * N);
	
	
	cudaFree(G_A);
	cudaFree(G_B);
	cudaFree(G_C);
	cudaFree(G_C1);
	cudaFree(G_C2);
	free(C_A);
	free(C_B);
	free(C_C);
	free(C_C1);
	cudaDeviceReset();
}

int main(){
    printf("100x100 matrix addition.\n");
	computeMatrix(100);
	
	//printf("200x200 matrix addition.\n");
	//computeMatrix(200);
	
	//printf("500x500 matrix addition.\n");
	//computeMatrix(500);
	
	//printf("1000x1000 matrix addition.\n");
	//computeMatrix(1000);
	
	//printf("1500x1500 matrix addition.\n");
	//computeMatrix(1500);
	
	//printf("5000x5000 matrix addition.\n");
	//computeMatrix(14200);
    return 0;
}
