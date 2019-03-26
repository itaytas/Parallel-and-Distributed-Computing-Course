#include <stdlib.h>
#include <stdio.h>
#include "io.h"
#include "kMeans.h"
#include "quality.h"

#define INITIAL_K 2
#define MASTER 0
#define FLAG_CONTINUE 1
#define FLAG_STOP 0
#define DATA_FROM_FILE_SIZE 4


int main(int argc, char *argv[])
{
	//variables required for MPI initialization
	int numprocs, myid;

	//variables for program
	int i, j,
		k,							// k is actual num of clusters
		toContonueKMeans,			// 1 = CONTINUE, 0 = STOP
		numOfDims,
		numOfAllVectors,
		maxNumOfClusters,			    // required maximum number of clusters to find stated in input file
		iterationLimit,
		*vectorIndexOfCluster,  //[numOfAllVectors] in use by 'A' only! index is vector and value is cluster index which the vector												belongs to
		*vectorIndexOfClusterSingleMachine,		//[numVectorInMachine] in use for the K-Means only!

		*sendCounts = NULL,
		*recvCounts = NULL,
		*displsScatter = NULL,
		*displsGather = NULL,
		numVectorsInMachine, 
		*dataFromFile;

	double	 *vectorsReadFile,
		*devVectors = NULL,			//pointer to vectors on GPU
									//*vectorGroup,			//[sizeOfCluster] in use by p0 only!! contains all vectors that belong to a certain cluster
		*diameters,			//[numClusters] in use by p0 only. contains diameters for each cluster
		**clusters,
		requiredQuality,			// required quality to reach stated in input file
		computedQuality,
		**vectorsInEveryMachine;			//[numVectorsInMachine][numOfDims]

	char    **productNames = NULL,
		*inputFile = "D:\\Sales_train_with_spaces.txt",
		*outputFile = "D:\\Itay_Taasiri_KMeans_data_result.txt";

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Status status;

	k = INITIAL_K; //initial k as stated by project req.
	toContonueKMeans = FLAG_CONTINUE; //continue FLAG iterations

	sendCounts = (int*)malloc(numprocs * sizeof(int));
	assert(sendCounts != NULL);
	displsScatter = (int*)malloc(numprocs * sizeof(int));
	assert(displsScatter != NULL);
	recvCounts = (int*)malloc(numprocs * sizeof(int));
	assert(recvCounts != NULL);
	displsGather = (int*)malloc(numprocs * sizeof(int));
	assert(displsGather != NULL);
	dataFromFile = (int*)malloc(DATA_FROM_FILE_SIZE * sizeof(int));
	assert(dataFromFile != NULL);

	// Measuring Time - START
	double t1 = MPI_Wtime();

	if (myid == 0)
	{
		
	    
		
		//read vectors from data-set file
		vectorsReadFile = readVectorsFromFile(inputFile, &numOfDims, &numOfAllVectors, &maxNumOfClusters,
			&iterationLimit, &requiredQuality, productNames);

		//vectorIndexOfCluster - the cluster id for each vector
		vectorIndexOfCluster = (int*)malloc(numOfAllVectors * sizeof(int));
		assert(vectorIndexOfCluster != NULL);


		dataFromFile[0] = numOfAllVectors;
		dataFromFile[1] = numOfDims;
		dataFromFile[2] = iterationLimit;
		dataFromFile[3] = maxNumOfClusters;
	}

	//broadcasting helpful data from file to all procs
	MPI_Bcast(dataFromFile, DATA_FROM_FILE_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&requiredQuality, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	numOfAllVectors = dataFromFile[0];
	numOfDims = dataFromFile[1];
	iterationLimit = dataFromFile[2];
	maxNumOfClusters = dataFromFile[3];

	//compute the chunck of vectors each machine gets
	calculateVectorsForScatter(numOfAllVectors, numprocs, numOfDims, sendCounts, displsScatter);

	//make arrangements to gather all the vector to cluster relevancies
	for (i = 0; i < numprocs; ++i)
	{
		recvCounts[i] = sendCounts[i] / numOfDims;
	}

	displsGather[0] = 0;

	for (i = 1; i < numprocs; ++i)
	{
		displsGather[i] = displsGather[i - 1] + recvCounts[i - 1];
	}

	//sendCount[myid] is number of doubles every machine gets
	numVectorsInMachine = sendCounts[myid] / numOfDims;

	//allocate memory for storing vectors on each machine
	vectorsInEveryMachine = (double**)malloc(numVectorsInMachine * sizeof(double*));
	assert(vectorsInEveryMachine != NULL);
	vectorsInEveryMachine[0] = (double*)malloc(numVectorsInMachine * numOfDims * sizeof(double));
	assert(vectorsInEveryMachine[0] != NULL);

	for (i = 1; i < numVectorsInMachine; ++i)
	{
		vectorsInEveryMachine[i] = vectorsInEveryMachine[i - 1] + numOfDims;
	}

	//scatter vectors to all machines
	MPI_Scatterv(vectorsReadFile, sendCounts, displsScatter, MPI_DOUBLE, vectorsInEveryMachine[0],
		sendCounts[myid], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	copyVectorsToGPU(vectorsInEveryMachine, &devVectors, numVectorsInMachine, numOfDims);

	/*
	* At this point 'A' holds ALL vectors from the data-set in *vectorsReadFile.

	* Each machine (including 'A') has the vectors it was assigned in the load balance, stored in *vectorsInEveryMachine.

	* 'A' initialize the **clusters with the first 'k' vectors from *vectorsReadFile.

	* Next up is a loop, where kMeans will be called with ascend k's, and produce an array of relevance of vectors to clusters.

	* 'A' compute the quality for k clusters relying on *diameters which define the max diameter of every cluster.

	* When the quality will meet the requirements, or kMeans will be called with the
	maximum number of clusters, the program will terminate.
	*/

	do // All machines
	{
		//allocating the clusters matrix
		clusters = (double**)malloc(k * sizeof(double*));
		assert(clusters != NULL);
		clusters[0] = (double*)malloc(k * numOfDims * sizeof(double));
		assert(clusters[0] != NULL);
		
		//vectorIndexOfClusterSingleMachine - the cluster id for each vector in each machine
		vectorIndexOfClusterSingleMachine = (int*)malloc(numVectorsInMachine * sizeof(int));
		assert(vectorIndexOfClusterSingleMachine != NULL);
		
		for (i = 1; i < k; ++i)
		{
			clusters[i] = clusters[i - 1] + numOfDims;
		}

		// 'A' picks the first k elements in *vectorsReadFile as initial cluster centers
		if (myid == 0 && k <= numOfAllVectors)
		{
			for (i = 0; i < k; ++i)
			{
				for (j = 0; j < numOfDims; ++j)
				{
					clusters[i][j] = vectorsReadFile[j + i * numOfDims];
				}
			}
		}

		// 'A' shares the initialized clusters with all machines
		MPI_Bcast(clusters[0], k * numOfDims, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// ################# start the computation #################
		k_means(vectorsInEveryMachine, devVectors, numOfDims, numVectorsInMachine, k, iterationLimit, vectorIndexOfClusterSingleMachine, clusters, MPI_COMM_WORLD);

		MPI_Gatherv(vectorIndexOfClusterSingleMachine, numVectorsInMachine, MPI_INT, vectorIndexOfCluster,
			recvCounts, displsGather, MPI_INT, 0, MPI_COMM_WORLD);

		//computing cluster quality
		if (myid == 0)
		{
			// *diameters defines the MAX diameter of every cluster
			diameters = computeClusterDiameter(vectorsReadFile, numOfAllVectors, k, numOfDims, vectorIndexOfCluster);

			computedQuality = computeClusterQuality(clusters, k, numOfDims, diameters);

			free(diameters);
		}

		MPI_Bcast(&computedQuality, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		if ((computedQuality > requiredQuality) && (k < maxNumOfClusters))
		{
			free(clusters[0]);
			free(clusters);
			free(vectorIndexOfClusterSingleMachine);
			k++;
		}
		else { toContonueKMeans = FLAG_STOP; }

	} while (toContonueKMeans == FLAG_CONTINUE);

	// 'A' writes the result into a file in location: "D:\\Itay_Taasiri_KMeans_data_result.txt"
	if (myid == 0)
	{
		writeClustersToFile(outputFile, clusters, k, numOfDims, computedQuality);
	}

	// Measuring Time - START
	double t2 = MPI_Wtime() - t1;

	//free all memory allocated
	FreeVectorsOnGPU(&devVectors);
	free(clusters[0]);
	free(clusters);
	free(vectorIndexOfClusterSingleMachine);

	if (myid == 0)
	{
		free(vectorIndexOfCluster);
		printf("time=%.5f quality=%.5f\n\n", t2, computedQuality);
	}

	MPI_Finalize();
	return 0;
}
