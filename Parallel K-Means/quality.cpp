#include "quality.h"
#include <stdio.h>
#include <mpi.h>

// dimension = numOfDims , vector2 and vector1 are arrays with size of [numdimensions]
double euclidDistanceForQuality(int dimension, double *vector1, double *vector2)
{
	int i;
	double dist = 0.0;

	for (i = 0; i < dimension; ++i)
	{
		dist += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i]);
	}

	return sqrt(dist);
}


double* computeClusterDiameter(double *vectors,
	int    numOfVectors, int numOfClusters, int numOfDims,
	int    *vectorIndexOfCluster)
{
	double diameter, dist, *diametersThreads, *diameters;
	int i, j, numThreads, tid, offset;

	diameter = 0.0;

	numThreads = omp_get_max_threads();

	diametersThreads = (double*)calloc(numThreads * numOfClusters, sizeof(double));

	diameters = (double*)malloc(numOfClusters * sizeof(double));

#pragma omp parallel for private(j, tid, dist, offset) shared(diametersThreads)
	for (i = 0; i < numOfVectors; ++i)
	{
		tid = omp_get_thread_num();
		offset = tid * numOfClusters;

		for (j = i + 1; j < numOfVectors; ++j)
		{
			if (vectorIndexOfCluster[i] == vectorIndexOfCluster[j])
			{
				dist = euclidDistanceForQuality(numOfDims, vectors + (i * numOfDims), vectors + (j * numOfDims));
				if (dist > diametersThreads[offset + vectorIndexOfCluster[i]])
					diametersThreads[offset + vectorIndexOfCluster[i]] = dist;
			}
		}

	}
	//seq
	//t0 computes max of diameters
	for (i = 0; i < numOfClusters; i++)
	{
		diameters[i] = diametersThreads[i];
		for (j = 1; j < numThreads; j++)
		{
			if (diameters[i] < diametersThreads[j * numOfClusters + i])
				diameters[i] = diametersThreads[j * numOfClusters + i];
		}
	}
	return diameters;
}

double computeClusterQuality(double **clusters, int numOfClusters, int numOfDims, double *diameters)
{
	int i, j, numElements = 0;
	double quality = 0.0;
#pragma omp parallel for private(j) reduction(+ : quality)
	for (i = 0; i < numOfClusters; ++i)
	{
		for (j = i + 1; j < numOfClusters; ++j)
		{
			quality += (diameters[i] + diameters[j]) / euclidDistanceForQuality(numOfDims, clusters[i], clusters[j]);
			numElements++;
		}
	}
	return quality / numElements;
}