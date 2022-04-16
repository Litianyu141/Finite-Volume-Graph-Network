#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//CUDA RunTime API
#include <cuda_runtime.h>

#define DATA_SIZE 1048576

int data[DATA_SIZE];
void printDeviceProp(const cudaDeviceProp& prop)
{
	printf("Device Name : %s.\n", prop.name);
	printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
	printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
	printf("regsPerBlock : %d.\n", prop.regsPerBlock);
	printf("warpSize : %d.\n", prop.warpSize);
	printf("memPitch : %d.\n", prop.memPitch);
	printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("totalConstMem : %d.\n", prop.totalConstMem);
	printf("major.minor : %d.%d.\n", prop.major, prop.minor);
	printf("clockRate : %d.\n", prop.clockRate);
	printf("textureAlignment : %d.\n", prop.textureAlignment);
	printf("deviceOverlap : %d.\n", prop.deviceOverlap);
	printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

void GenerateNumbers(int* number, int size)    //产生大量0-9之间的随机数
{
	for (int i = 0; i < size; i++) {
		number[i] = rand() % 10;
	}
}

//CUDA 初始化
bool InitCUDA()
{
	int count;

	//取得支持Cuda的装置的数目
	cudaGetDeviceCount(&count);

	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;

	for (i = 0; i < count; i++) {

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		//打印设备信息
		printDeviceProp(prop);

		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}

	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}


// __global__ 函数 (GPU上执行) 计算立方和
__global__ static void sumOfSquares(int* num, int* result, clock_t* time)
{
	int sum = 0;

	int i;

	clock_t start = clock();

	for (i = 0; i < DATA_SIZE; i++) {

		sum += num[i] * num[i] * num[i];

	}

	*result = sum;

	*time = clock() - start;

}





int main()
{

	//CUDA 初始化
	if (!InitCUDA()) {
		return 0;
	}

	//生成随机数
	GenerateNumbers(data, DATA_SIZE);

	/*把数据复制到显卡内存中*/
	int* gpudata, * result;

	clock_t* time;

	//cudaMalloc 取得一块显卡内存 ( 其中result用来存储计算结果，time用来存储运行时间 )
	cudaMalloc((void**)&gpudata, sizeof(int) * DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int));
	cudaMalloc((void**)&time, sizeof(clock_t));

	//cudaMemcpy 将产生的随机数复制到显卡内存中
	//cudaMemcpyHostToDevice - 从内存复制到显卡内存
	//cudaMemcpyDeviceToHost - 从显卡内存复制到内存
	cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

	// 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
	sumOfSquares << <1, 1, 0 >> > (gpudata, result, time);


	/*把结果从显示芯片复制回主内存*/

	int sum;
	clock_t time_used;

	//cudaMemcpy 将结果从显存中复制回内存
	cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);

	//Free
	cudaFree(gpudata);
	cudaFree(result);
	cudaFree(time);

	printf("GPUsum: %d time: %d\n", sum, time_used);

	sum = 0;

	for (int i = 0; i < DATA_SIZE; i++) {
		sum += data[i] * data[i] * data[i];
	}

	printf("CPUsum: %d \n", sum);

	system("pause");

	return 0;
}
