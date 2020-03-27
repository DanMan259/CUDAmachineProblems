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
		C[idx] = 0.0;
		for (int i = 0; i < N; i++)
			C[idx] += A[row * N + i] * B[i * N + col];
	}
}

__global__ void MatrixMulGPUSingle(float* A, float* B, float* C, const int N) {
	for (int row = 0; row < sDim; row++) {
		for (int col = 0; col < sDim; col++) {
			if (row < sDim && col < sDim) {
				C[row * sDim + col] = 0.0;
				for (int k = 0; k < sDim; k++)
					C[row * sDim + col] += A[row * sDim + k] * B[k * sDim + col];
			}
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
			for (int k = 0; k < N; k++) 
				C[i * N + j] += A[i * N + k] * B[k * N + j];
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

float GPUtest(float* C_A, float* C_B, float* CPUResult, const int blockSize, const int N){
	// Initialize variables
	cudaEvent_t gStart, gEnd;
	float timeDuration;
	float *G_A, *G_B, *G_C, *GPUResult;
	size_t size = N * N * sizeof(float);
	
	// Initialize GPU variables
	cudaMalloc((void**)&G_A, size);
	cudaMalloc((void**)&G_B, size);
	cudaMalloc((void**)&G_C, size);
	memset(GPUResult, 0.0, size);
	cudaEventCreate(&gStart);
    cudaEventCreate(&gEnd);
	
	// Copy over the data
	cudaMemcpy(G_A, C_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(G_B, C_B, size, cudaMemcpyHostToDevice);
	
	// Create block
	int numBlocks = sDim / blockSize;
	if (N % blockSize) numBlocks++;
	dim3 block(blockWidth, blockWidth, 1);
	dim3 grid(numBlocks, numBlocks, 1);
	
	// Perform GPU comparison
	cudaEventRecord(gStart);
	if (blockSize == 0)
		MatrixMulGPUSingle <<<1, 1>>> (G_A, G_B, G_C, N);
	else
		MatrixMulGPU <<<grid, block>>> (G_A, G_B, G_C, N);
	cudaEventRecord(gEnd);
	cudaEventSynchronize(gEnd);
	cudaEventElapsedTime(&timeDuration, gStart, gEnd);
	
	cudaMemcpy(GPUResult, G_C, size, cudaMemcpyDeviceToHost);
	checkResult(CPUResult, GPUResult, N);
	
	cudaFree(G_A);
	cudaFree(G_B);
	cudaFree(G_C);
	free(GPUResult);
	return (timeDuration/1000.0);
}

void computeMatrix(const int N) {
	// Initial prints
	printf("------------------------------------------------------------------------\n\n");
	printf("%dx%d matrix multiplication.\n\n", N, N, blockSize);

	// Initialize Host variables
	float *C_A, *C_B, *C_C;
	size_t size = N * N * sizeof(float);
	FILE *fp;
	
	// Initialize space
	C_A = (float*)malloc(size);
	C_B = (float*)malloc(size);
	C_C = (float*)malloc(size);
	fp=fopen("machineProblem3.csv","a");
	
	// Set with random data
	initialData(C_A, N);
	initialData(C_B, N);
	memset(C_C, 0.0, size);

	// Serial Test CPU
	auto cStart = std::chrono::high_resolution_clock::now();
	MatrixMulCPU(C_A, C_B, C_C, N);
	auto cEnd = std::chrono::high_resolution_clock::now();
	auto timeElapse = std::chrono::duration_cast<std::chrono::seconds>(cEnd - cStart).count();
	printf("The CPU took %d to perform the computation.\n\n", timeElapse);
	fprintf(fp,"%d,CPU,%d,%d\n",N, blockSize,timeElapse);
	
	// Test Complete parallel Computation
	int blockSizes [] = {0, 2, 4, 10, 20, 25};
	float timeDuration;
	
	for (int i = 0; i < 6; j++){
		timeDuration = GPUtest(C_A, C_B, C_C, blockSizes[i]);
		printf("The GPU took %f to perform the computation with block size %d.\n", timeDuration, blockSizes[i]);
		fprintf(fp,"%d,GPU,%d,%f\n",N,blockSizes[i],timeDuration);
	}
	
	// Free all the memory
	free(C_A);
	free(C_B);
	free(C_C);
	fclose(fp);
	cudaDeviceReset();
}

int main(){
	FILE *fp;
	fp=fopen("machineProblem3.csv","w");
	fprintf(fp,"matrixSize,processor,blockSize,time\n");
	fclose(fp);
	int matrixWidths [] = {100, 200, 500, 1000, 1500, 5000};
	
	for (int i = 0; i < 6; i++){
		computeMatrix(matrixWidths[i]);

	printf("------------------------------------------------------------------------\n\n");
    
	return 0;
}
