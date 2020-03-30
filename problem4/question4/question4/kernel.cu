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

__global__ void TiledMatrixMulGPU(float* A, float* B, float* C, const int N, const int tileWidth) {
	__shared__ float t_A [tileWidth][tileWidth];
	__shared__ float t_B [tileWidth][tileWidth];
	
	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.x;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int row = by * blockDim.y + ty;
	unsigned int col = bx * blockDim.x + tx;
	
	float cValue = 0.0;
	for (int i = 0; i < (N/tileWidth); i++){
		t_A[ty][tx] = A[row*N + i*tileWidth+tx];
		t_B[ty][tx] = B[(i*tileWidth+ty)*N + col];	
		__syncthreads();
		for(int j = 0; j < (tileWidth); j++)
			cValue += t_A[ty][j] *t_B[j][tx];
		__syncthreads();
	}
	C[row*N+col] = cValue;
}

void initialData(float* matrix, const int size){
	for (int i = 0; i < size; i++)
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

void checkResult(float* CPU, float* GPU, const int size) {
	double epsilon = 1.0E-8;
	
	for (int i = 0; i < size; i++){
		if (abs(CPU[i] - GPU[i]) > epsilon){
			printf("CPU %f GPU %f ", CPU[i], GPU[i]);
			printf("Arrays do not match.\n\n");
			return;
		}
	}
	printf("Test PASSED\n\n");
}

void printArr(float* matrix, const int size) {
	printf("[");
	for (int i = 0; i < size; i++)
		printf("%f,", matrix[i]);
	printf("\b]\n");

}

void GPUtest(float* C_A, float* C_B, float* CPUResult, const int blockSize, const int N){
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

	// Create block
	int numBlocks = N / blockSize;
	if (N % blockSize) numBlocks++;
	dim3 block(blockSize, blockSize, 1);
	dim3 grid(numBlocks, numBlocks, 1);

	cudaEventRecord(gStart);
	TiledMatrixMulGPU <<<grid, block>>> (G_A, G_B, G_C, N, blockSize); // MIGHT CHANGE EVERYTHING TO TILE SIZE
	cudaEventRecord(gEnd);
	
	cudaEventSynchronize(gEnd);
	cudaEventElapsedTime(&timeDuration, gStart, gEnd);
	
	cudaMemcpy(GPUResult, G_C, size, cudaMemcpyDeviceToHost);
	checkResult(CPUResult, GPUResult, N*N);
	
	cudaFree(G_A);
	cudaFree(G_B);
	cudaFree(G_C);
	free(GPUResult);
	
	FILE *fp;
	fp=fopen("machineProblem4.csv","a");
	printf("The GPU took %f to perform the computation with block size %d.\n", timeDuration, blockSizes[i]);
	fprintf(fp,"%d,GPU,%d,%f\n",N,blockSize,timeDuration);
	fclose(fp);	

}

void computeMatrix(const int N) {
	// Initial prints
	printf("------------------------------------------------------------------------\n\n");
	printf("%dx%d matrix multiplication.\n\n", N, N);

	// Initialize Host variables
	float *C_A, *C_B, *C_C;
	size_t size = N * N * sizeof(float);
	
	// Initialize space
	C_A = (float*)malloc(size);
	C_B = (float*)malloc(size);
	C_C = (float*)malloc(size);
	
	// Set with random data
	initialData(C_A, N*N);
	initialData(C_B, N*N);
	memset(C_C, 0.0, size);

	// Serial Test CPU
	auto cStart = std::chrono::high_resolution_clock::now();
	MatrixMulCPU(C_A, C_B, C_C, N);
	auto cEnd = std::chrono::high_resolution_clock::now();
	auto timeElapse = (std::chrono::duration_cast<std::chrono::microseconds>(cEnd - cStart).count())/1000.0;
	printf("The CPU took %f to perform the computation.\n\n", timeElapse);
	
	// Write to file
	FILE *fp;
	fp=fopen("machineProblem4.csv","a");
	fprintf(fp,"%d,CPU,0,%f\n",N,timeElapse);
	fclose(fp);
	
	// Test Complete parallel Computation
	int blockSizes [] = {2, 4, 10, 20, 25};
	float timeDuration;
	
	for (int i = 0; i < 5; i++)
		GPUtest(C_A, C_B, C_C, blockSizes[i], N);
	
	// Free all the memory
	free(C_A);
	free(C_B);
	free(C_C);
	cudaDeviceReset();
}

// --------------------BONUS-----------------------------

__global__ void TiledMatrixMulGPUBonus(float* A, float* B, float* C, const int M, const int N, const int K, const int tileWidth) {

	__shared__ float t_A [tileWidth][tileWidth];
	__shared__ float t_B [tileWidth][tileWidth];
	
	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.x;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int row = by * blockDim.y + ty;
	unsigned int col = bx * blockDim.x + tx;
	
	float cValue = 0.0;
	for (int i = 0; i < ((tileWidth + N - 1) / tileWidth); i++){
		if (i*tileWidth + tx < N && row < M) {
			t_A[ty][tx] = A[row*N + i*tileWidth + tx];
		} else {
			t_A[ty][tx] = 0;
		}
		
		if ((k*tileWidth + ty < N) && col < K){
			t_B[ty][tx] = B[(i*tileWidth+ty)*K + col];
		} else {
			t_B[ty][tx] = 0;
		}
		__syncthreads();
		for(int j = 0; j < tileWidth; j++)
			cValue += t_A[ty][j] *t_B[j][tx];
		__syncthreads();
	}
	
	if (row < M && col < K) 
		C[row*K+col] = cValue;
}


void computeMatrixBonus(const int M, const int N, const int K) {
	// Initial prints
	printf("------------------------------------------------------------------------\n\n");
	printf("%dx%d and %dx%d matrix multiplication.\n\n", M, N, N, K);

	float *C_A, *C_B, *C_C, *GPUResult;
	size_t sizeA = M * N * sizeof(float);
	size_t sizeB = N * K * sizeof(float);
	size_t sizeC = M * K * sizeof(float);
	
	// Initialize space
	C_A = (float*)malloc(sizeA);
	C_B = (float*)malloc(sizeB);
	C_C = (float*)malloc(sizeC);
	GPUResult = (float*)malloc(sizeC);
	
	// Set with random data
	initialData(C_A, M * N);
	initialData(C_B, N * K);
	memset(C_C, 0.0, sizeC);
	memset(GPUResult, 0.0, sizeC);

	// Serial Test CPU
	MatrixMulCPU(C_A, C_B, C_C, N);
	
	// GPU calculations
	float *G_A, *G_B, *G_C, *GPUResult;
	
	// Initialize GPU variables
	cudaMalloc((void**)&G_A, sizeA);
	cudaMalloc((void**)&G_B, sizeB);
	cudaMalloc((void**)&G_C, sizeC);
	// Copy over values
	cudaMemcpy(G_A, C_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(G_B, C_B, size, cudaMemcpyHostToDevice);
	
	
	dim3 block(8, 8);
	dim3 thread((int)ceil((K + block.x - 1)) / block.x, (int)ceil((M + block.y - 1) / block.y));
	
	// Test different tile sizes
	TiledMatrixMulGPUBonus <<<grid, block >>> (G_A, G_B, G_C, M, N, K, 8);
	cudaMemcpy(GPUResult, G_C, size, cudaMemcpyDeviceToHost);
	checkResult(CPUResult, GPUResult, N*N);
	
	TiledMatrixMulGPUBonus <<<grid, block >>> (G_A, G_B, G_C, M, N, K, 14);
	cudaMemcpy(GPUResult, G_C, size, cudaMemcpyDeviceToHost);
	checkResult(CPUResult, GPUResult, N*N);
	

	// Free all the memory
	free(C_A);
	free(C_B);
	free(C_C);
	free(GPUResult);
	cudaFree(G_A);
	cudaFree(G_B);
	cudaFree(G_C);
	cudaDeviceReset();
}

// ------------------------------------------------------


int main(){
	FILE *fp;
	fp=fopen("machineProblem4.csv","w");
	fprintf(fp,"matrixSize,processor,tileSize,time\n");
	fclose(fp);
	int matrixWidths [] = {100, 200, 500, 1000, 1500, 5000};
	
	for (int i = 0; i < 6; i++)
		computeMatrix(matrixWidths[i]);

	printf("------------------------------------------------------------------------\n\n");
    
	printf("BONUS\n");
	
	computeMatrixBonus(250, 300, 450);
	
	return 0;
}
