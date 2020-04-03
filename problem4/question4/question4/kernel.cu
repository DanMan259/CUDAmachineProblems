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

__global__ void TiledMatrixMulGPU2(float* A, float* B, float* C, const int N) {
	__shared__ float t_A [2][2];
	__shared__ float t_B [2][2];

	unsigned int tileWidth = 2;
	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int row = by * tileWidth + ty;
	unsigned int col = bx * tileWidth + tx;

	float cValue = 0.0;
	for (int i = 0; i < (N / tileWidth); i++) {
		t_A[ty][tx] = A[row*N + i*tileWidth + tx];
		t_B[ty][tx] = B[(i*tileWidth + ty)*N + col];
		__syncthreads();
		for (int j = 0; j < tileWidth; j++)
			cValue += t_A[ty][j] * t_B[j][tx];
		__syncthreads();
	}
	C[row*N + col] = cValue;
}

__global__ void TiledMatrixMulGPU4(float* A, float* B, float* C, const int N) {
	__shared__ float t_A [4][4];
	__shared__ float t_B [4][4];

	unsigned int tileWidth = 4;
	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int row = by * tileWidth + ty;
	unsigned int col = bx * tileWidth + tx;

	float cValue = 0.0;
	for (int i = 0; i < (N / tileWidth); i++) {
		t_A[ty][tx] = A[row*N + i*tileWidth + tx];
		t_B[ty][tx] = B[(i*tileWidth + ty)*N + col];
		__syncthreads();
		for (int j = 0; j < tileWidth; j++)
			cValue += t_A[ty][j] * t_B[j][tx];
		__syncthreads();
	}
	C[row*N + col] = cValue;
}	


__global__ void TiledMatrixMulGPU10(float* A, float* B, float* C, const int N) {
	__shared__ float t_A [10][10];
	__shared__ float t_B [10][10];

	unsigned int tileWidth = 10;
	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int row = by * tileWidth + ty;
	unsigned int col = bx * tileWidth + tx;

	float cValue = 0.0;
	for (int i = 0; i < (N / tileWidth); i++) {
		t_A[ty][tx] = A[row*N + i*tileWidth + tx];
		t_B[ty][tx] = B[(i*tileWidth + ty)*N + col];
		__syncthreads();
		for (int j = 0; j < tileWidth; j++)
			cValue += t_A[ty][j] * t_B[j][tx];
		__syncthreads();
	}
	C[row*N + col] = cValue;
}

__global__ void TiledMatrixMulGPU20(float* A, float* B, float* C, const int N) {
	__shared__ float t_A [20][20];
	__shared__ float t_B [20][20];

	unsigned int tileWidth = 20;
	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int row = by * tileWidth + ty;
	unsigned int col = bx * tileWidth + tx;

	float cValue = 0.0;
	for (int i = 0; i < (N / tileWidth); i++) {
		t_A[ty][tx] = A[row*N + i*tileWidth + tx];
		t_B[ty][tx] = B[(i*tileWidth + ty)*N + col];
		__syncthreads();
		for (int j = 0; j < tileWidth; j++)
			cValue += t_A[ty][j] * t_B[j][tx];
		__syncthreads();
	}
	C[row*N + col] = cValue;
}

__global__ void TiledMatrixMulGPU25(float* A, float* B, float* C, const int N, const int tileWidth) {
	__shared__ float t_A [25][25];
	__shared__ float t_B [25][25];

	unsigned int tileWidth = 25;
	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int row = by * tileWidth + ty;
	unsigned int col = bx * tileWidth + tx;

	float cValue = 0.0;
	for (int i = 0; i < (N / tileWidth); i++) {
		t_A[ty][tx] = A[row*N + i*tileWidth + tx];
		t_B[ty][tx] = B[(i*tileWidth + ty)*N + col];
		__syncthreads();
		for (int j = 0; j < tileWidth; j++)
			cValue += t_A[ty][j] * t_B[j][tx];
		__syncthreads();
	}
	C[row*N + col] = cValue;
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

void GPUtest(float* C_A, float* C_B, float* CPUResult, const int tileSize, const int N){
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
	dim3 block(tileSize, tileSize, 1);
	dim3 grid(N/tileSize, N/tileSize, 1);

	cudaEventRecord(gStart);
	if (tileSize == 2)
		TiledMatrixMulGPU2 <<<grid, block>>> (G_A, G_B, G_C, N, tileSize);
	else if (tileSize == 4)
		TiledMatrixMulGPU4 <<<grid, block>>> (G_A, G_B, G_C, N, tileSize);
	else if (tileSize == 10)
		TiledMatrixMulGPU10 <<<grid, block>>> (G_A, G_B, G_C, N, tileSize);
	else if (tileSize == 20)
		TiledMatrixMulGPU20 <<<grid, block>>> (G_A, G_B, G_C, N, tileSize);
	else if (tileSize == 25)
		TiledMatrixMulGPU25 <<<grid, block>>> (G_A, G_B, G_C, N, tileSize);
	cudaEventRecord(gEnd);
	
	cudaEventSynchronize(gEnd);
	cudaEventElapsedTime(&timeDuration, gStart, gEnd);
	
	printArr(GPUResult, N*N);
	cudaMemcpy(GPUResult, G_C, size, cudaMemcpyDeviceToHost);
	checkResult(CPUResult, GPUResult, N*N);
	
	cudaFree(G_A);
	cudaFree(G_B);
	cudaFree(G_C);
	free(GPUResult);
	
	FILE *fp;
	fp=fopen("machineProblem4.csv","a");
	printf("The GPU took %f to perform the computation with tile size %d.\n", timeDuration, tileSize);
	fprintf(fp,"%d,%d,%f\n",N,tileSize,timeDuration);
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
	MatrixMulCPU(C_A, C_B, C_C, N);

	printArr(C_A, N*N);
	printArr(C_B, N*N);
	printArr(C_C, N*N);
	
	// Test Complete parallel Computation
	int tileSizes [] = {2, 4, 10, 20, 25};
	
	for (int i = 4; i < 5; i++)
		GPUtest(C_A, C_B, C_C, tileSizes[i], N);
	
	// Free all the memory
	free(C_A);
	free(C_B);
	free(C_C);
	cudaDeviceReset();
}

// --------------------BONUS-----------------------------

#define BONUSTILE_1C 8
#define BONUSTILE_1R 8
#define BONUSTILE_2C 14
#define BONUSTILE_2R 14

__global__ void TiledMatrixMulGPUBonus(float* A, float* B, float* C, const int M, const int N, const int K, const int tileCase) {
	
	if (tileCase == 1){
		__shared__ float t_A [BONUSTILE_1R][BONUSTILE_1C];
		__shared__ float t_B [BONUSTILE_1R][BONUSTILE_1C];

		unsigned int bx = blockIdx.x;
		unsigned int by = blockIdx.x;
		unsigned int tx = threadIdx.x;
		unsigned int ty = threadIdx.y;
		unsigned int row = by * blockDim.y + ty;
		unsigned int col = bx * blockDim.x + tx;

		float cValue = 0.0;
		for (int i = 0; i < ((BONUSTILE_1C + N - 1) / BONUSTILE_1C); i++) {
			if (i*BONUSTILE_1C + tx < N && row < M) {
				t_A[ty][tx] = A[row*N + i*BONUSTILE_1C + tx];
			}
			else {
				t_A[ty][tx] = 0;
			}

			if ((i*BONUSTILE_1C + ty < N) && col < K) {
				t_B[ty][tx] = B[(i*BONUSTILE_1C + ty)*K + col];
			}
			else {
				t_B[ty][tx] = 0;
			}
			__syncthreads();
			for (int j = 0; j < BONUSTILE_1C; j++)
				cValue += t_A[ty][j] * t_B[j][tx];
			__syncthreads();
		}

		if (row < M && col < K)
			C[row*K + col] = cValue;
	} else {
		__shared__ float t_A [BONUSTILE_2R][BONUSTILE_2C];
		__shared__ float t_B [BONUSTILE_2R][BONUSTILE_2C];

		unsigned int bx = blockIdx.x;
		unsigned int by = blockIdx.x;
		unsigned int tx = threadIdx.x;
		unsigned int ty = threadIdx.y;
		unsigned int row = by * blockDim.y + ty;
		unsigned int col = bx * blockDim.x + tx;

		float cValue = 0.0;
		for (int i = 0; i < ((BONUSTILE_2C + N - 1) / BONUSTILE_2C); i++) {
			if (i*BONUSTILE_2C + tx < N && row < M) {
				t_A[ty][tx] = A[row*N + i*BONUSTILE_2C + tx];
			}
			else {
				t_A[ty][tx] = 0;
			}

			if ((i*BONUSTILE_2C + ty < N) && col < K) {
				t_B[ty][tx] = B[(i*BONUSTILE_2C + ty)*K + col];
			}
			else {
				t_B[ty][tx] = 0;
			}
			__syncthreads();
			for (int j = 0; j < BONUSTILE_2C; j++)
				cValue += t_A[ty][j] * t_B[j][tx];
			__syncthreads();
		}

		if (row < M && col < K)
			C[row*K + col] = cValue;
	
	}
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
	float *G_A, *G_B, *G_C;
	
	// Initialize GPU variables
	cudaMalloc((void**)&G_A, sizeA);
	cudaMalloc((void**)&G_B, sizeB);
	cudaMalloc((void**)&G_C, sizeC);
	// Copy over values
	cudaMemcpy(G_A, C_A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(G_B, C_B, sizeB, cudaMemcpyHostToDevice);
	
	dim3 block(8, 8);
	dim3 thread((int)ceil((K + block.x - 1)) / block.x, (int)ceil((M + block.y - 1) / block.y));
	
	// Test different tile sizes
	
	// Case 1 8x8
	dim3 block1(BONUSTILE_1R, BONUSTILE_1C);
	dim3 grid1((int)ceil((K + block.x - 1)) / block.x, (int)ceil((M + block.y - 1) / block.y));
	
	TiledMatrixMulGPUBonus <<<grid1, block1>>> (G_A, G_B, G_C, M, N, K, 1);
	cudaMemcpy(GPUResult, G_C, sizeC, cudaMemcpyDeviceToHost);
	checkResult(C_C, GPUResult, N*N);
	
	// Case 2 14x14
	dim3 block2(BONUSTILE_2R, BONUSTILE_2C);
	dim3 grid2((int)ceil((K + block.x - 1)) / block.x, (int)ceil((M + block.y - 1) / block.y));
	
	TiledMatrixMulGPUBonus <<<grid2, block2>>> (G_A, G_B, G_C, M, N, K, 2);
	cudaMemcpy(GPUResult, G_C, sizeC, cudaMemcpyDeviceToHost);
	checkResult(C_C, GPUResult, N*N);
	

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
	fprintf(fp,"matrixSize,tileSize,time\n");
	fclose(fp);
	int matrixWidths [] = {2, 100, 200, 500, 1000, 1500, 5000};
	
	for (int i = 0; i < 1; i++)
		computeMatrix(matrixWidths[i]);

	printf("------------------------------------------------------------------------\n\n");
    
	printf("BONUS\n");
	
	//computeMatrixBonus(250, 300, 450);
	
	return 0;
}
