#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



using namespace std;

__global__ void cube(long *deviceOutput, long *deviceInput)
{
	int idx = threadIdx.x;
	long f = deviceInput[idx];
	deviceOutput[idx] = f * f * f;
}

int main()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	
	const int ArraySize = 1024;
	const int ArrayBytes = ArraySize * sizeof(long);
	
	//ArrayBytes = 4 * ArraySize.  Because the dIn and dOut are both floating 
	//Numbers, meaning I need to know the size to allocate when doing Malloc
	//And Memcpy.  
	//cout << ArrayBytes << endl;

	

	long hostInput[ArraySize];
	
	for (int i = 0; i < ArraySize; i++) {
		hostInput[i] = i;
	}

	long hostOutput[ArraySize];

	//GPU memory pointers
	long * deviceInput;
	long * deviceOutput;

	

	//Allocate GPU memory
	cudaMalloc((void **)&deviceInput, ArrayBytes);
	cudaMalloc((void **)&deviceOutput, ArrayBytes);

	//Transfer the array to GPU
	cudaMemcpy(deviceInput, hostInput, ArrayBytes, cudaMemcpyHostToDevice);

	//Launch the Kernal
	//This Kernal has 1 Thread Block, and that thread block has ArraySize amount of Threads.
	cudaEventRecord(start);
	cube<<<2, ArraySize >>>(deviceOutput, deviceInput);
	cudaEventRecord(stop);
	//Copy back result from GPU to CPU

	cudaMemcpy(hostOutput, deviceOutput, ArrayBytes, cudaMemcpyDeviceToHost);

	for (int i = 0; i < ArraySize; i++) {
		cout << hostOutput[i] << endl;
	}
	

	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	cout << "Milliseconds: " << ms << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//End of program.
	
	return 0;
}
