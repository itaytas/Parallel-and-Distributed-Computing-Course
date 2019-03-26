#ifndef __K_MEANS_H
#define __K_MEANS_H

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include "kernel.h"

void calculateVectorsForScatter(int  numOfAllVectors, int  numOfMachines, int  numOfDims,
	int *sendCounts, int *displs);

void k_means(double **vectors,
	double *devVectors,											//in: [numOfVectors][numOfDims]
	int        numOfDims, int numOfVectors, int numOfClusters,
	int        limit,   			 							//max num of iterations	
	int       *vectorToClusterRelevance,						//out: [numOfVectors] 
	double    **clusters,    									//out: [numOfClusters][numOfDims] 
	MPI_Comm   comm);

cudaError_t copyVectorsToGPU(double **vectors, double **devVectors, int numOfVectors, int numOfDims);

cudaError_t FreeVectorsOnGPU(double **devVectors);




#endif // !__K_MEANS_H