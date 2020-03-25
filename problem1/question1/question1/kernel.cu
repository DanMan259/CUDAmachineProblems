/*
	Name: Daniyal Manair
	Student Number: 20064993
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <string>

void printDeviceProp(int devIdx){
	cudaDeviceProp devInfo;
	cudaGetDeviceProperties(&devInfo, devIdx);
	printf("----------------------------------------------\n");
	printf("Device %d: %s\n", devIdx, devInfo.name);
	printf("\tClock Rate: %d kHz\n", devInfo.clockRate);
	printf("\tStreaming Multiprocessors: %d\n", devInfo.multiProcessorCount);
	printf("\tNumber of Cores: %d\n", (_ConvertSMVer2Cores(devInfo.major, devInfo.minor)*devInfo.multiProcessorCount));
	printf("\tWarp Size: %d\n", devInfo.warpSize);
	printf("\tAmount of Global Memory: %zu\n", devInfo.totalGlobalMem);
	printf("\tAmount of Constant Memory: %zu\n", devInfo.totalConstMem);
	printf("\tAmount of Shared Memory per Block: %zu\n", devInfo.sharedMemPerBlock);
	printf("\tNumber of Registers per Block: %d\n", devInfo.regsPerBlock);
	printf("\tMaximum Number of Threads per Block: %d\n", devInfo.maxThreadsPerBlock);

	printf("\tMaximum Size of each dimension of a Block:\n");
	for (int j = 0; j < 3; j++) {
		printf("\t\tMaximum dimension size of dimension %d: %d\n", j, devInfo.maxThreadsDim[j]);
	}

	printf("\tMaximum Size of each dimension of a Grid:\n");
	for (int j = 0; j < 3; j++) {
		printf("\t\tMaximum grid size of dimension %d: %d\n", j, devInfo.maxGridSize[j]);
		}
	printf("----------------------------------------------\n");
}

int main(void){
    int deviceCount;
	
	cudaGetDeviceCount(&deviceCount);
	
	for (int i = 0; i < deviceCount; i++){
		printDeviceProp(i);
	}
	
	return 0;
}

