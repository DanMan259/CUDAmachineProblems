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

__global__ void TiledMatrixMulGPU(float* A, float* B, float* C, const int M, const int N, const int K) {
	__shared__ float t_A [TILE_WIDTH][TILE_WIDTH];
	__shared__ float t_B [TILE_WIDTH][TILE_WIDTH];
	
	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.x;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int row = by * blockDim.y + ty;
	unsigned int col = bx * blockDim.x + tx;
	
	float cValue = 0.0;
	for (int i = 0; i < (N/TILE_WIDTH); i++){
		t_A[ty][tx] = A[row*N + i*TILE_WIDTH+tx];
		t_B[ty][tx] = B[(i*TILE_WIDTH+ty)*K + col];	
		__syncthreads();
		for(int j = 0; j < (TILE_WIDTH); j++)
			cValue += t_A[ty][j] *t_B[j][tx];
		__syncthreads();
	}
	C[row*K+col] = cValue;
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
	GPUResult = (float*)malloc(size);
	memset(GPUResult, 0.0, size);
	cudaEventCreate(&gStart);
    cudaEventCreate(&gEnd);
	
	// Copy over the data
	cudaMemcpy(G_A, C_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(G_B, C_B, size, cudaMemcpyHostToDevice);
	
	// Perform GPU comparison
	if (blockSize == 0){
		cudaEventRecord(gStart);
		MatrixMulGPUSingle <<<1, 1>>> (G_A, G_B, G_C, N);
		cudaEventRecord(gEnd);
	} else {
		// Create block
		int numBlocks = N / blockSize;
		if (N % blockSize) numBlocks++;
		dim3 block(blockSize, blockSize, 1);
		dim3 grid(numBlocks, numBlocks, 1);

		cudaEventRecord(gStart);
		MatrixMulGPU <<<grid, block >>> (G_A, G_B, G_C, N);
		cudaEventRecord(gEnd);
	}
		
	
	cudaEventSynchronize(gEnd);
	cudaEventElapsedTime(&timeDuration, gStart, gEnd);
	
	cudaMemcpy(GPUResult, G_C, size, cudaMemcpyDeviceToHost);
	checkResult(CPUResult, GPUResult, N);
	
	cudaFree(G_A);
	cudaFree(G_B);
	cudaFree(G_C);
	free(GPUResult);
	return timeDuration;
}

void computeMatrix(const int N) {
	// Initial prints
	printf("------------------------------------------------------------------------\n\n");
	printf("%dx%d matrix multiplication.\n\n", N, N);

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
	auto timeElapse = (std::chrono::duration_cast<std::chrono::microseconds>(cEnd - cStart).count())/1000.0;
	printf("The CPU took %f to perform the computation.\n\n", timeElapse);
	fprintf(fp,"%d,CPU,0,%f\n",N,timeElapse);
	
	// Test Complete parallel Computation
	int blockSizes [] = {0, 2, 4, 10, 20, 25};
	float timeDuration;
	
	for (int i = 0; i < 6; i++){
		timeDuration = GPUtest(C_A, C_B, C_C, blockSizes[i], N);
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
	
	for (int i = 0; i < 6; i++)
		computeMatrix(matrixWidths[i]);

	printf("------------------------------------------------------------------------\n\n");
    
	return 0;
}
