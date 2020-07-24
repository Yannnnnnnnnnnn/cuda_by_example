#include <iostream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cuda.h"
#include "driver_types.h"
#include <windows.h>

#include <cuda_runtime.h>
#include "helper_cuda.h"
//#include <opencv2/highgui.hpp>

texture<uchar, 1, cudaReadModeElementType> texRef;

__global__ void helloWorld()
{
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	printf("helloWorld %d \n", threadId);
}

__global__ void gaussianCUDAKernal(float *gaussianKernal, float*src_d,float* dst_d, int width,int height)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < 1 || i >= height - 2 || j < 1 || j >= width - 2)
		return;

	dst_d[i*width + j] = 
		src_d[(i - 1)*width + j - 1] * gaussianKernal[0] +
		src_d[(i - 1)*width + j] * gaussianKernal[1] +
		src_d[(i - 1)*width + j + 1] * gaussianKernal[2] +
		src_d[(i)*width + j - 1] * gaussianKernal[3] +
		src_d[(i)*width + j] * gaussianKernal[4] +
		src_d[(i)*width + j + 1] * gaussianKernal[5] +
		src_d[(i + 1)*width + j - 1] * gaussianKernal[6] +
		src_d[(i + 1)*width + j] * gaussianKernal[7] +
		src_d[(i + 1)*width + j + 1] * gaussianKernal[8];
}

__global__ void kernelUseTex(float *gaussianKernal, float*src_d, float* dst_d, int width, int height)
{
	int j = threadIdx.x + blockIdx.x*blockDim.x;
	int i = threadIdx.y + blockIdx.y*blockDim.y;
	if (i < 1 || i >= height - 2 || j < 1 || j >= width - 2)
		return;

	dst_d[i*width + j] =
		tex1Dfetch( texRef, (i - 1)*width + j - 1) * gaussianKernal[0] +
		tex1Dfetch(	texRef,(i - 1)*width + j) * gaussianKernal[1] +
		tex1Dfetch(	texRef,(i - 1)*width + j + 1) * gaussianKernal[2] +
		tex1Dfetch(	texRef,(i)*width + j - 1) * gaussianKernal[3] +
		tex1Dfetch(	texRef,(i)*width + j) * gaussianKernal[4] +
		tex1Dfetch(	texRef,(i)*width + j + 1) * gaussianKernal[5] +
		tex1Dfetch(	texRef,(i + 1)*width + j - 1) * gaussianKernal[6] +
		tex1Dfetch(	texRef,(i + 1)*width + j) * gaussianKernal[7] +
		tex1Dfetch(	texRef,(i + 1)*width + j + 1) * gaussianKernal[8];
}


int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void main()
{
	cv::Mat img = cv::imread("E:/project/third_party/opencv-3.4.3/samples/data/lena.jpg");
	//cv::Mat img = cv::imread("E:/project/Image_accelerate/data/genOrthophoto/original_images/0713ER039012.jpg");
	//cv::resize(img, img, cv::Size(), 20, 20);


	cv::Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	gray.convertTo(gray, CV_32FC1, 1 / 255.0);
	cv::Mat dst(gray);

	//cv::namedWindow("lena");
	//cv::imshow("lena", gray);
	//cv::waitKey(0);

	int width = gray.cols;
	int height = gray.rows;


	const float gaussianKernal[9] =
	{
		0.1110185185345953f, 0.1111573784521313f, 0.1110185185345953f,
	    0.1111573784521313f, 0.1112964120530937f, 0.1111573784521313f,
	    0.1110185185345953f, 0.1111573784521313f, 0.1110185185345953f
	};

	//cv::Mat kernelX = cv::getGaussianKernel(3, 20);
	//cv::Mat kernelY = cv::getGaussianKernel(3, 20);
	//cv::Mat G = kernelX * kernelY.t();
	//std::cout << G << std::endl;


	// CPU Version
	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	
	// 1. 申请CPU内存
	float* src_h = new float[width * height];
	float* dst_h = new float[width * height];

	// 2. 初始化内存
	for (int i = 0; i < width*height; ++i)
	{
		src_h[i] = ((float*)gray.data)[i];
	}

	// 3. 计算高斯滤波
	QueryPerformanceCounter(&t1);
	for (int i = 0; i < 100; ++i)
	{
		for (int i = 0; i < height; ++i)
		{
			if (i < 1 || i >= height - 2)
				continue;
			for (int j = 0; j < width; ++j)
			{
				if (j < 1 || j >= width - 2)
					continue;
				int _i = i;
				int _j = j;

				dst_h[i*width + j] =
					src_h[(i - 1)*width + j - 1] * gaussianKernal[0] +
					src_h[(i - 1)*width + j] * gaussianKernal[1] +
					src_h[(i - 1)*width + j + 1] * gaussianKernal[2] +
					src_h[(i)*width + j - 1] * gaussianKernal[3] +
					src_h[(i)*width + j] * gaussianKernal[4] +
					src_h[(i)*width + j + 1] * gaussianKernal[5] +
					src_h[(i + 1)*width + j - 1] * gaussianKernal[6] +
					src_h[(i + 1)*width + j] * gaussianKernal[7] +
					src_h[(i + 1)*width + j + 1] * gaussianKernal[8];
			}
		}
	}
	QueryPerformanceCounter(&t2);

	// 4. show result
	//memcpy(dst.data, dst_h, width * height * sizeof(float));
	//cv::namedWindow("result");
	//cv::imshow("result",dst);
	//cv::waitKey(0);

	
	std::cout << "cpu cost: " << (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart << std::endl;
	double cpuTime = (t2.QuadPart - t1.QuadPart)*1.0;

	// GPU version 
	{

		// Warm up
		helloWorld << <1, 5 >> > ();

		// 1. 申请GPU内存
		int size = width * height * sizeof(float);
		float* dst_d = new float[width * height];
		float* src_d = new float[width * height];
		float* gaussianKernal_d;

		checkCudaErrors( cudaMalloc((void **)&dst_d, size ));
		checkCudaErrors( cudaMalloc((void **)&src_d, size ));
		checkCudaErrors( cudaMalloc((void **)&gaussianKernal_d, 9 * sizeof(4)));

		// 2. 拷贝主机内存到设备
		checkCudaErrors(cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(gaussianKernal_d, gaussianKernal,9*sizeof(4), cudaMemcpyHostToDevice));

		// 3. 计算kernel
		// Global memory
		{
			QueryPerformanceCounter(&t1);

			for (int i = 0; i < 100; ++i)
			{
				//dim3 block (32, 32);
				//dim3 grid  (iDivUp(width, block.x), iDivUp(height, block.y));
				dim3 block(8, 128);
				dim3 grid(iDivUp(height, block.x), iDivUp(width, block.y));
				gaussianCUDAKernal << <grid, block >> > (gaussianKernal_d, src_d, dst_d, width, height);

				checkCudaErrors(cudaDeviceSynchronize());
				checkCudaErrors(cudaGetLastError());
			}

			QueryPerformanceCounter(&t2);

		}

		// Texture memory
		//{
		//	QueryPerformanceCounter(&t1);
		//	for (int i = 0; i < 100; ++i)
		//	{
		//		checkCudaErrors(cudaBindTexture(NULL, texRef, src_d, size));

		//		dim3 block(32, 32);
		//		dim3 grid(iDivUp(width, block.x), iDivUp(height, block.y));
		//		kernelUseTex << <grid, block >> > (gaussianKernal_d, src_d, dst_d, width, height);

		//		checkCudaErrors(cudaDeviceSynchronize());
		//		checkCudaErrors(cudaGetLastError());
		//	}
		//	QueryPerformanceCounter(&t2);
		//}

		// 4. 拷贝设备到主机
		checkCudaErrors(cudaMemcpy(dst.data, dst_d, size, cudaMemcpyDeviceToHost));

		//cv::namedWindow("result1");
		//cv::imshow("result1", dst);
		//cv::waitKey(0);


		// 5. 释放设备内存
		cudaFree(dst_d);
		cudaFree(src_d);
		cudaFree(gaussianKernal_d);


	}

	// 验证结果  &&&&
	int count(0);
	for (int i = 0; i < width*height; ++i)
	{
		if (dst_h[i] - *((float*)(dst.data) + i) > 10e-6)
		{
			count++;
			std::cout << "wrong !" << *((float*)(dst.data) + i) << "  " << dst_h[i] << std::endl;
		}
	}
	std::cout << "wrong count: " << count << std::endl;

	std::cout << "gpu cost: " << (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart << std::endl;
	double gpuTime = (t2.QuadPart - t1.QuadPart)*1.0;

	std::cout << "加速比: " << cpuTime / gpuTime << std::endl;


	free(src_h);
	free(dst_h);

	return;
}