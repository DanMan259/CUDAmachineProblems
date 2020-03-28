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

__global__ void sumMatrixGPU(float* A, float* B, float* C, const int N) {
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idx = row * N + col;
	if (row < N && col < N){
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

void computeMatrix(const int N) {
	// Initial prints
	printf("------------------------------------------------------------------------\n\n");
	printf("%dx%d matrix addition.\n\n", N, N);

	// Initialize Host variables
	float* C_A, *C_B, *C_C, *C_C1;
	float timeDuration;
	size_t size = N * N * sizeof(float);
	cudaEvent_t gStart, gEnd;
	FILE *fp;
	
	// Initialize space
	C_A = (float*)malloc(size);
	C_B = (float*)malloc(size);
	C_C = (float*)malloc(size);
	C_C1 = (float*)malloc(size);
	fp=fopen("machineProblem2.csv","a");
	cudaEventCreate(&gStart);
    cudaEventCreate(&gEnd);
	
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
	sumMatrixCPU(C_A, C_B, C_C, N);
	auto end = std::chrono::high_resolution_clock::now();
	auto timeElapse = (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())/1000.0;
	printf("The CPU took %f to perform the computation.\n\n", timeElapse);
	fprintf(fp,"%d,CPU,ELEMENT,0,%f\n",N,timeElapse);
	
	// Test Complete parallel Computation
	dim3 block(16, 16);
	dim3 thread((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
	
	cudaEventRecord(gStart);
	sumMatrixGPU <<<thread, block >>> (G_A, G_B, G_C, N);
	cudaEventRecord(gEnd);
	cudaEventSynchronize(gEnd);
	cudaEventElapsedTime(&timeDuration, gStart, gEnd);
	printf("The GPU took %f to perform the computation with one thread per element.\n", timeDuration);
	fprintf(fp,"%d,GPU,ELEMENT,16,%f\n",N,timeDuration);
	
	// Copy over the result and compare
	cudaMemcpy(C_C1, G_C, size, cudaMemcpyDeviceToHost);
	checkResult(C_C, C_C1, N);
	
	// Test row based parallel Computation
	dim3 block1(16);
	dim3 thread1((N + block1.x - 1) / block1.x);
	
	cudaEventRecord(gStart);
	sumMatrixGPUperRow <<<thread1, block1 >>> (G_A, G_B, G_C1, N);
	cudaEventRecord(gEnd);
	cudaEventSynchronize(gEnd);
	cudaEventElapsedTime(&timeDuration, gStart, gEnd);
	printf("The GPU took %f to perform the computation with one thread per Row.\n", timeDuration);
	fprintf(fp,"%d,GPU,ROW,16,%f\n",N,timeDuration);

	// Copy over the result and compare
	cudaMemcpy(C_C1, G_C1, size, cudaMemcpyDeviceToHost);
	checkResult(C_C, C_C1, N);

	// Test Complete parallel Computation
	dim3 block2(16);
	dim3 thread2((N + block2.x - 1) / block2.x);
	
	// Test column based parallel Computation
	cudaEventRecord(gStart);
	sumMatrixGPUperCol <<<thread2, block2 >>> (G_A, G_B, G_C2, N);
	cudaEventRecord(gEnd);
	cudaEventSynchronize(gEnd);
	cudaEventElapsedTime(&timeDuration, gStart, gEnd);
	printf("The GPU took %f to perform the computation with one thread per Column.\n", timeDuration);
	fprintf(fp,"%d,GPU,COL,16,%f\n",N,timeDuration);
	
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
	fclose(fp);
	cudaDeviceReset();
}

int main(){
	FILE *fp;
	fp=fopen("machineProblem2.csv","w");
	fprintf(fp,"matrixSize,processor,type,blockSize,time\n");
	fclose(fp);
	computeMatrix(100);
	computeMatrix(200);
	computeMatrix(500);
	computeMatrix(1000);
	computeMatrix(1500);
	computeMatrix(3000);
	computeMatrix(5000);
	printf("------------------------------------------------------------------------\n\n");


    return 0;
}
