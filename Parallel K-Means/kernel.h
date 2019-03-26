#ifndef __KERNEL_H_
#define __KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void computeDistancesArray(double *devVectors, double *devClusters,
	int numOfVectors, int numThreadsInBlock, int numOfDims,
	double *devDistsVectorsToClusters);

__global__ void findMinDistanceForEachVectorFromCluster(int numOfVectors, int numOfClusters, int numThreadsInBlock,
	double *devDistsVectorsToClusters,
	int   *devVectorIndexOfCluster);

cudaError_t computeClustersMeansWithCUDA(double *devVectors, double **clusters,
	int numOfVectors, int numOfClusters, int numOfDims,
	int *vectorIndexOfCluster);

#endif // !__KERNEL_H_
