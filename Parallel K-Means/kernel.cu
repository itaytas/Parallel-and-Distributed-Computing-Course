#include "kernel.h"

/*
* NOTICE: numThreads = numVectors.
* Each thread is a vector. computes the distances between his vector and all clusters.
* devDistsVectorsToClusters[numThreads * numClusters].
* EXAMPLE: devDistsVectorsToClusters[0]-[numClusters] contains all distances from v1 to all clusters
*/
__global__ void computeDistancesArray(double *devVectors, double *devClusters,
	int numOfVectors, int numThreadsInBlock, int numOfDims,
	double *devDistsVectorsToClusters)
{
	int i, blockID = blockIdx.x;
	double result = 0;

	if (blockID == gridDim.x - 1 && numOfVectors % blockDim.x <= threadIdx.x)
		return;

	for (i = 0; i < numOfDims; ++i)
	{
		result += (devVectors[(blockID*numThreadsInBlock + threadIdx.x)*numOfDims + i] - devClusters[threadIdx.y*numOfDims + i]) *  (devVectors[(blockID*numThreadsInBlock + threadIdx.x)*numOfDims + i] - devClusters[threadIdx.y*numOfDims + i]);
	}
	devDistsVectorsToClusters[numOfVectors*threadIdx.y + (blockID*numThreadsInBlock + threadIdx.x)] = result;
}


/*
* NOTICE: numThreads = numVectors.
* Each thread is a vector. Traverses devDistsVectorsToClusters[] and finds the min distance.
* Writes it to devVectorIndexOfCluster[numVectors]
*/
__global__ void findMinDistanceForEachVectorFromCluster(int numOfVectors, int numOfClusters, int numThreadsInBlock,
	double *devDistsVectorsToClusters,
	int   *devVectorIndexOfCluster)
{
	int i, xid = threadIdx.x, blockId = blockIdx.x;
	double minIndex = 0, minDistance, tempDistance;

	if (blockIdx.x == gridDim.x - 1 && numOfVectors % blockDim.x <= xid)
		return;

	minDistance = devDistsVectorsToClusters[numThreadsInBlock*blockId + xid];

	for (i = 1; i < numOfClusters; i++)
	{
		tempDistance = devDistsVectorsToClusters[numThreadsInBlock*blockId + xid + i*numOfVectors];
		if (minDistance > tempDistance)
		{
			minIndex = i;
			minDistance = tempDistance;
		}
	}
	devVectorIndexOfCluster[numThreadsInBlock*blockId + xid] = minIndex;
}

cudaError_t computeClustersMeansWithCUDA(double *devVectors, double **clusters,
	int numOfVectors, int numOfClusters, int numOfDims,
	int *vectorIndexOfCluster)
{
	cudaError_t cudaStatus;
	cudaDeviceProp devProp;

	int maxThreadsPerBlock, maxGridSize[3];
	int numBlocks, numThreadsInBlock;

	cudaGetDeviceProperties(&devProp, 0); // 0 is device 0

	for (int i = 0; i < 3; ++i)
		maxGridSize[i] = devProp.maxGridSize[i];

	//configuring kerenl params
	numThreadsInBlock = devProp.maxThreadsPerBlock / numOfClusters;
	dim3 dim(numThreadsInBlock, numOfClusters);
	numBlocks = numOfVectors / numThreadsInBlock;

	if (numOfVectors % numThreadsInBlock > 0) { numBlocks++; }

	double *devClusters;
	double *devDistsVectorsToClusters = 0;
	int   *devVectorIndexOfCluster = 0;

	// Allocate GPU buffers for three vectors (two input, one output) 
	cudaStatus = cudaMalloc((void**)&devClusters, numOfClusters*numOfDims * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&devDistsVectorsToClusters, numOfClusters*numOfVectors * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&devVectorIndexOfCluster, numOfVectors * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy input from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(devClusters, clusters[0], numOfClusters*numOfDims * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//launch kernels//
	computeDistancesArray << <numBlocks, dim >> > (devVectors, devClusters, numOfVectors, numThreadsInBlock, numOfDims, devDistsVectorsToClusters);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	/* cudaDeviceSynchronize waits for the kernel to finish, and returns
	any errors encountered during the launch*/
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	numThreadsInBlock = devProp.maxThreadsPerBlock;
	numBlocks = numOfVectors / numThreadsInBlock;
	if (numOfVectors % numThreadsInBlock > 0) { numBlocks++; }

	findMinDistanceForEachVectorFromCluster << <numBlocks, numThreadsInBlock >> > (numOfVectors, numOfClusters, numThreadsInBlock, devDistsVectorsToClusters, devVectorIndexOfCluster);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	/* cudaDeviceSynchronize waits for the kernel to finish, and returns
	any errors encountered during the launch*/
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(vectorIndexOfCluster, devVectorIndexOfCluster, numOfVectors * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(devClusters);
	cudaFree(devDistsVectorsToClusters);
	cudaFree(devVectorIndexOfCluster);

	return cudaStatus;
}
