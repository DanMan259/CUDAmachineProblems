/*
	Name: Daniyal Manair
	Student Number: 20064993
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <stdio.h>
#include <random>
#include <algorithm>
#include <chrono>
#include <map>

__global__ void MatrixMulGPU(float* A, float* B, float* C, const int N) {
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idx = row * N + col;
	if (row < N && col < N){
		for (int i = 0; i < N; i++)
			C[row * N + col] += A[row * N + i] * B[i * N + col];
	}
}

__global__ void MatrixMulGPUperRow(float* A, float* B, float* C, const int N) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < N){
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++)
				C[row * N + i] += A[row * N + j] * B[j * N + i];
		}
	}
}

__global__ void MatrixMulGPUperCol(float* A, float* B, float* C, const int N) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < N){
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++)
				C[i * N + col] += A[i * N + j] * B[j * N + col];
		}
	}
}

void initialData(float* matrix, const int N){
	for (int i = 0; i < (N*N); i++)
		matrix[i] = (float)(rand() & 0xFF) / 10.0f;
}

void MatrixMulCPU(float* A, float* B, float* C, const int N){
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int p = 0; p < N; p++) {
				C[i * N + j] += A[i * N + p] * B[p * N + j];
			}
		}
	}
}

void checkResult(float* CPU, float* GPU, const int N) {
	double epsilon = 1.0E-8;
	
	for (int i = 0; i < (N*N); i++){
		if (abs(CPU[i] - GPU[i]) > epsilon){
			printf("CPU %f GPU %f ", CPU[i], GPU[i]);
			printf("Arrays do not match.\n\n");
			return;
		}
	}
	printf("Test PASSED\n\n");
}

void printArr(float* matrix, const int N) {
	printf("[");
	for (int i = 0; i < (N*N); i++)
		printf("%f,", matrix[i]);
	printf("\b]\n");

}

void computeMatrix(const int N, const int blockSize) {
	// Initial prints
	printf("------------------------------------------------------------------------\n\n");
	printf("%dx%d matrix addition with block size %d.\n\n", N, N, blockSize);

	// Initialize Host variables
	float* C_A, *C_B, *C_C, *C_C1;
	size_t size = N * N * sizeof(float);
	
	// Initialize space
	C_A = (float*)malloc(size);
	C_B = (float*)malloc(size);
	C_C = (float*)malloc(size);
	C_C1 = (float*)malloc(size);

	// Set with random data
	initialData(C_A, N);
	initialData(C_B, N);
	memset(C_C, 0, N);
	memset(C_C1, 0, N);

	// Initialize GPU variables
	float* G_A, *G_B, *G_C, *G_C1, *G_C2;
	cudaMalloc((void**)&G_A, size);
	cudaMalloc((void**)&G_B, size);
	cudaMalloc((void**)&G_C, size);
	cudaMalloc((void**)&G_C1, size);
	cudaMalloc((void**)&G_C2, size);
	
	// Copy over the data
	cudaMemcpy(G_A, C_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(G_B, C_B, size, cudaMemcpyHostToDevice);

	// Serial Test CPU
	auto start = std::chrono::high_resolution_clock::now();
	MatrixMulCPU(C_A, C_B, C_C, N);
	auto end = std::chrono::high_resolution_clock::now();
	auto timeElapse = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("The CPU took %d to perform the computation.\n\n", timeElapse);
	
	
	// Test Complete parallel Computation
	dim3 block(blockSize, blockSize);
	dim3 thread((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
	
	start = std::chrono::high_resolution_clock::now();
	cudaDeviceSynchronize();
	MatrixMulGPU <<<thread, block>>> (G_A, G_B, G_C, N);
	cudaDeviceSynchronize();
	end = std::chrono::high_resolution_clock::now();
	timeElapse = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("The GPU took %d to perform the computation with one thread per element.\n", timeElapse);
	
	// Copy over the result and compare
	cudaMemcpy(C_C1, G_C, size, cudaMemcpyDeviceToHost);
	checkResult(C_C, C_C1, N);
	
	// Test row based parallel Computation
	dim3 block(blockSize);
	dim3 thread((N + block.x - 1) / block.x);
	
	start = std::chrono::high_resolution_clock::now();
	cudaDeviceSynchronize();
	MatrixMulGPUperRow <<<thread1, block1 >>> (G_A, G_B, G_C1, N);
	cudaDeviceSynchronize();
	end = std::chrono::high_resolution_clock::now();
	timeElapse = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("The GPU took %d to perform the computation with one thread per Row.\n", timeElapse);

	// Copy over the result and compare
	cudaMemcpy(C_C1, G_C1, size, cudaMemcpyDeviceToHost);
	checkResult(C_C, C_C1, N);

	// Test Complete parallel Computation
	dim3 block(blockSize);
	dim3 thread((N + block.x - 1) / block.x);
	
	// Test column based parallel Computation
	start = std::chrono::high_resolution_clock::now();
	cudaDeviceSynchronize();
	MatrixMulGPUperCol <<<thread2, block2 >>> (G_A, G_B, G_C2, N);
	cudaDeviceSynchronize();
	end = std::chrono::high_resolution_clock::now();
	timeElapse = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("The GPU took %d to perform the computation with one thread per Column.\n", timeElapse);
	
	// Copy over the result and compare
	cudaMemcpy(C_C1, G_C2, size, cudaMemcpyDeviceToHost);
	checkResult(C_C, C_C1, N);
	
	// Free all the memory
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
	int matrixWidths [] = {100, 200, 500, 1000, 1500, 5000};
	int blockSizes [] = {2, 4, 10, 20, 25};
	for (int i = 0; i < 6; i++{
		for (int j = 0; i < 5; j++){
			computeMatrix(matrixWidths[i], blockSize[j]);
		}
	}
	printf("------------------------------------------------------------------------\n\n");
    return 0;
}
