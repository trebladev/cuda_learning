#include <cuda_runtime.h>
#include <stdio.h>
#include <utils.h>

int recursiveReduce(int *data, int const size)
{
  // terminate check
  if (size == 1) return data[0];
  // renew the stride
  int const stride = size / 2;
  if (size % 2 == 1)
  {
    for (int i=0;i<stride;++i)
    {
      data[i] += data[i+stride];
    }
    data[0] += data[size-1];
  }
  else
  {
    for (int i=0;i<stride;++i)
    {
      data[i] += data[i+stride];
    }
  }
  // call
  return recursiveReduce(data, stride);
}

__global__ void warmup(int *g_idata, int *g_odata, unsigned int n)
{
  // set thread include
  unsigned int tid = threadIdx.x;
  // boundary check
  if (tid >= n) return;
  // convert global data pointer to the data
  int *idata = g_idata + blockIdx.x*blockDim.x;
  // in-place reduction in global memory
  for (int stride=1;stride < blockDim.x; stride *=2)
  {
    if((tid%(2*stride))==0)
    {
      idata[tid] += idata[tid+stride];
    }
    // synchronize within blockDim
    __syncthreads();
  }
  // write result for this block to global memory
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}
__global__ void reduceNeighbored(int * g_idata, int * g_odata, unsigned int n)
{
  // set thread include
  unsigned int tid = threadIdx.x;
  unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
  // boundary check
  if (tid >= n) return;
  // convert global data pointer to this block
  int *idata = g_idata + blockIdx.x*blockDim.x;
  // in-place reduction in global memory
  for (int stride=1;stride < blockDim.x; stride *=2)
  {
    // convert tid into local array index
    if ((tid%(2*stride)) == 0)
    {
      idata[tid] += idata[tid+stride];
    }
    // synchronize within blockDim
    __syncthreads();
  }
  // write result for this block to global memory
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLess(int * g_idata,int *g_odata,unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	// convert global data pointer to the local point of this block
	int *idata = g_idata + blockIdx.x*blockDim.x;
	if (idx > n)
		return;
	//in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		//convert tid into local array index
		int index = 2 * stride *tid;
		if (index < blockDim.x)
		{
			idata[index] += idata[index + stride];
		}
		__syncthreads();
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
{

  unsigned int tid = threadIdx.x;
  unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
  // convert global data point to the local point for this block 
  int *idata = g_idata + blockIdx.x*blockDim.x;
  if (idx > n) return;
  // in-place reduction in global memory
  for (int stride=blockDim.x/2; stride > 0; stride>>=1)
  {
    if (tid<stride)
    {
      idata[tid] += idata[tid+stride];
    }
    __syncthreads();
  }
  // write result for this block to global menory
  if (tid==0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceShare(int *g_idata, int *g_odata, unsigned int n)
{
  __shared__ float idata[1024];

  unsigned int tid = threadIdx.x;
  unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
  // convert global data point to the local point for this block 
  // int *idata = g_idata + blockIdx.x*blockDim.x;
  
  idata[tid] = g_idata[idx];
  if (idx > n) return;
  // in-place reduction in global memory
  for (int stride=blockDim.x/2; stride > 0; stride>>=1)
  {
    if (tid<stride)
    {
      idata[tid] += idata[tid+stride];
    }
    __syncthreads();
  }
  // write result for this block to global menory
  if (tid==0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceIdle(int *g_idata, int *g_odata, unsigned int n)
{
  __shared__ float idata[512];

  unsigned int tid = threadIdx.x;
  unsigned idx = blockIdx.x*blockDim.x*2+threadIdx.x;
  // convert global data point to the local point for this block 
  // int *idata = g_idata + blockIdx.x*blockDim.x;
  
  idata[tid] = g_idata[idx]+g_idata[idx+blockDim.x];
  __syncthreads();
  if (idx > n) return;
  // in-place reduction in global memory
  for (int stride=blockDim.x/2; stride > 0; stride>>=1)
  {
    if (tid<stride)
    {
      idata[tid] += idata[tid+stride];
    }
    __syncthreads();
  }
  // write result for this block to global menory
  if (tid==0)
    g_odata[blockIdx.x] = idata[0];
}

__device__ void warpReduce(volatile float* cache, unsigned int tid)
{
  cache[tid]+=cache[tid+32];
  cache[tid]+=cache[tid+16];
  cache[tid]+=cache[tid+8];
  cache[tid]+=cache[tid+4];
  cache[tid]+=cache[tid+2];
  cache[tid]+=cache[tid+1];

}

__global__ void reduceWarp(int *g_idata, int *g_odata, unsigned int n)
{
  __shared__ float idata[512];

  unsigned int tid = threadIdx.x;
  unsigned idx = blockIdx.x*blockDim.x*2+threadIdx.x;
  // convert global data point to the local point for this block 
  // int *idata = g_idata + blockIdx.x*blockDim.x;
  
  idata[tid] = g_idata[idx]+g_idata[idx+blockDim.x];
  __syncthreads();
  if (idx > n) return;
  // in-place reduction in global memory
  for (int stride=blockDim.x/2; stride > 32; stride>>=1)
  {
    if (tid<stride)
    {
      idata[tid] += idata[tid+stride];
    }
    __syncthreads();
  }
  // write result for this block to global menory
  if (tid < 32) warpReduce(idata, tid);
  if (tid==0)
    g_odata[blockIdx.x] = idata[0];
}

template <unsigned int blockSize>
__device__ void warpReduceTemplate(volatile float* cache, int tid)
{
  if(blockSize >= 64) cache[tid]+=cache[tid+32];
  if(blockSize >= 32) cache[tid]+=cache[tid+16];
  if(blockSize >= 16) cache[tid]+=cache[tid+8];
  if(blockSize >= 8) cache[tid]+=cache[tid+4];
  if(blockSize >= 4) cache[tid]+=cache[tid+2];
  if(blockSize >= 2) cache[tid]+=cache[tid+1];
}

template <unsigned int blockSize>
__global__ void reduceWarpTemplate(int *g_idata, int *g_odata, unsigned int n)
{
  __shared__ float idata[512];

  unsigned int tid = threadIdx.x;
  unsigned idx = blockIdx.x*blockDim.x*2+threadIdx.x;
  // convert global data point to the local point for this block 
  // int *idata = g_idata + blockIdx.x*blockDim.x;
  
  idata[tid] = g_idata[idx]+g_idata[idx+blockDim.x];
  __syncthreads();
  if (idx > n) return;
  // in-place reduction in global memory
  if (blockSize>=512 && tid<256){
    idata[tid]+=idata[tid+256];
    __syncthreads();
  }
  if (blockSize>=256 && tid<128){
    idata[tid]+=idata[tid+128];
    __syncthreads();
  }
  if (blockSize>=128 && tid<64){
    idata[tid]+=idata[tid+64];
    __syncthreads();
  }

  // write result for this block to global menory
  if (tid < 32) warpReduceTemplate<blockSize>(idata, tid);
  if (tid==0)
    g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char** argv)
{
  initDevice(0);

  bool bResult = false;
  
  int size = 1<<24;
  printf("with array size %d\n", size);

  // execution configuation
  int blocksize = 1024;
  if (argc > 1)
    blocksize = atoi(argv[1]);

  dim3 block(blocksize, 1);
  dim3 block_half(blocksize/2, 1);
  dim3 grid((size-1)/block.x+1,1);
  printf("grid %d block %d \n", grid.x, block.x);

  // allocate host memory
  size_t bytes = size * sizeof(int);
  int *idata_host = (int*)malloc(bytes);
  int *odata_host = (int*)malloc(grid.x * sizeof(int));
  int *tmp = (int*)malloc(bytes);

  // initialize the array
  initialData_int(idata_host, size);

  memcpy(tmp, idata_host, bytes);
  double iStart, iElaps;
  int gpu_sum = 0;

  // device memory
  int * idata_dev = NULL;
  int * odata_dev = NULL;
  CHECK(cudaMalloc((void**)&idata_dev, bytes));
  CHECK(cudaMalloc((void**)&odata_dev, bytes));

  // cpu reduction
  int cpu_sum = 0;
  iStart = cpuSecond();
  
  for (int i=0;i<size;++i)
  {
    cpu_sum += tmp[i];
  }
  printf("cpu sum:%d\n", cpu_sum);
  iElaps = cpuSecond() - iStart;
  printf("cpu reduce elapsed %lf ms cpu_sum:%d\n", iElaps, cpu_sum);

  // kernel 1:reduceNeighbored
  CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  iStart = cpuSecond();
  warmup<<<grid, block>>>(idata_dev, odata_dev, size);
  cudaDeviceSynchronize();
  iElaps = cpuSecond()-iStart;
  cudaMemcpy(odata_host, odata_dev, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu warmup                 elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	//kernel 1:reduceNeighbored

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceNeighbored <<<grid, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceNeighbored       elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	//kernel 2:reduceNeighboredLess

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceNeighboredLess <<<grid, block>>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceNeighboredLess   elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	//kernel 3:reduceInterleaved
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceInterleaved <<<grid, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceInterleaved      elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

  //kernel 4:reduceShare
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceShare<<<grid, block>>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceShare            elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

  //kernel 5:reduceIdle
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceIdle<<<grid, block_half>>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceIdle             elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block_half.x);

  //kernel 6:reduceWarp
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceWarp<<<grid, block_half>>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceWarp             elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block_half.x);

  //kernel 7:reduceWarpTemplate
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceWarpTemplate<512><<<grid, block_half>>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceWarpTemplate     elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block_half.x);


	// free host memory

	free(idata_host);
	free(odata_host);
	CHECK(cudaFree(idata_dev));
	CHECK(cudaFree(odata_dev));

	//reset device
	cudaDeviceReset();

	//check the results
	if (gpu_sum == cpu_sum)
	{
		printf("Test success!\n");
	}
	return EXIT_SUCCESS;

}






