
#include "kMeans.h"

void calculateVectorsForScatter(int  numOfAllVectors, int  numOfMachines, int  numOfDims,
	int *sendCounts, int *displs)
{
	int i, remainder, index, *vectorCounterForMachine;

	vectorCounterForMachine = (int*)malloc(numOfMachines * sizeof(int));

	remainder = numOfAllVectors % numOfMachines;
	index = 0;

	for (i = 0; i < numOfMachines; ++i)
	{
		vectorCounterForMachine[i] = numOfAllVectors / numOfMachines;
		if (remainder > 0)
		{
			vectorCounterForMachine[i]++;
			remainder--;
		}

		sendCounts[i] = vectorCounterForMachine[i] * numOfDims;
		displs[i] = index;
		index += sendCounts[i];
	}
}


void k_means(double **vectors,
	double *devVectors,											//in: [numOfVectors][numOfDims]
	int        numOfDims, int numOfVectors, int numOfClusters,
	int        limit,   			 							//max num of iterations	
	int       *vectorToClusterRelevance,						//out: [numOfVectors] 
	double    **clusters,    									//out: [numOfClusters][numOfDims] 
	MPI_Comm   comm)

{
	int      i, j, index, numIterations = 0;
	int		 sumDelta = 0;   // the sum of all the machimes' delta indicates whether to stop the iteration
	int     *CUDAvectorIndexOfCluster;
	int     *newClusterSize; //[numOfClusters]: no. vectors assigned in each new cluster                             
	int     *clusterSize;    //[numOfClusters]: temp buffer for reduction 
	int      delta;            //num of vectors that change their cluster
	double  **newClusters;    //[numOfClusters][numOfDims]

							  //initialize vectorToClusterRelevance[]
	for (i = 0; i < numOfVectors; ++i)
	{
		vectorToClusterRelevance[i] = -1;
	}

	//initializing memory 
	CUDAvectorIndexOfCluster = (int*)malloc(numOfVectors * sizeof(int));
	assert(CUDAvectorIndexOfCluster != NULL);

	newClusterSize = (int*)calloc(numOfClusters, sizeof(int));
	assert(newClusterSize != NULL);

	clusterSize = (int*)calloc(numOfClusters, sizeof(int));
	assert(clusterSize != NULL);

	newClusters = (double**)malloc(numOfClusters * sizeof(double*));
	assert(newClusters != NULL);
	newClusters[0] = (double*)calloc(numOfClusters * numOfDims, sizeof(double));
	assert(newClusters[0] != NULL);
	for (i = 1; i < numOfClusters; ++i) //creates race conditions - must be seq!
	{
		newClusters[i] = newClusters[i - 1] + numOfDims;
	}

	//start the k-means iterations
	do
	{
		delta = 0;

		computeClustersMeansWithCUDA(devVectors, clusters, numOfVectors, numOfClusters, numOfDims, CUDAvectorIndexOfCluster);

		for (i = 0; i < numOfVectors; ++i)
		{

			if (vectorToClusterRelevance[i] != CUDAvectorIndexOfCluster[i])
			{
				delta++;
				vectorToClusterRelevance[i] = CUDAvectorIndexOfCluster[i];
			}

			/* find the array index of nearest cluster center */
			index = CUDAvectorIndexOfCluster[i];

			/* update new cluster centers : sum of objects located within */
			newClusterSize[index]++;
			for (j = 0; j < numOfDims; ++j)
				newClusters[index][j] += vectors[i][j];
		}

		MPI_Allreduce(&delta, &sumDelta, 1, MPI_INT, MPI_SUM, comm);

		if (sumDelta == 0)
		{
			break;
		}

		/* sum all data vectors in newClusters */
		MPI_Allreduce(newClusters[0], clusters[0], numOfClusters*numOfDims, MPI_DOUBLE, MPI_SUM, comm);
		MPI_Allreduce(newClusterSize, clusterSize, numOfClusters, MPI_INT, MPI_SUM, comm);

		/* average the sum and replace old cluster centers with newClusters */
		for (i = 0; i < numOfClusters; i++)
		{
			for (j = 0; j < numOfDims; j++)
			{
				if (clusterSize[i] > 1)
				{
					clusters[i][j] /= clusterSize[i];
				}
				newClusters[i][j] = 0.0;   /* set back to 0 */
			}
			newClusterSize[i] = 0;   /* set back to 0 */
		}

	} while (++numIterations < limit);

	//free all memory allocated
	free(newClusters[0]);
	free(newClusters);
	free(newClusterSize);
	free(clusterSize);
	free(CUDAvectorIndexOfCluster);
}

cudaError_t copyVectorsToGPU(double **vectors, double **devVectors, int numOfVectors, int numOfDims)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	cudaStatus = cudaMalloc((void**)devVectors, numOfVectors*numOfDims * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(devVectors);
	}

	cudaStatus = cudaMemcpy(*devVectors, vectors[0], numOfVectors*numOfDims * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(devVectors);
	}

	return cudaStatus;
}


cudaError_t FreeVectorsOnGPU(double **devVectors)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaFree(*devVectors);

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaFree failed!");
	}
	return cudaStatus;
}

