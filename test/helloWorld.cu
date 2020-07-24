#include <iostream>



__global__ void helloWorld()
{
	int threadId =  blockDim.x * blockIdx.x + threadIdx.x;
	printf("helloWorld %d \n", threadId);
}

void main()
{
	helloWorld << <2, 32 >> > ();
	return;
}