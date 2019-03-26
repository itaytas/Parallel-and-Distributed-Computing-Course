#ifndef __QUALITY_H_
#define __QUALITY_H_

#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>


// dimension = numOfDims , vector2 and vector1 are arrays with size of [numdimensions]
double euclidDistanceForQuality(int dimension, double *vector1, double *vector2);   	//[numdims]

																						//double* createClusterGroupArr(double *vectors,
																						//	int numOfVectors,
																						//	int clusterIndex,
																						//	int numOfDims,
																						//	int *vectorIndexOfCluster,
																						//	int *numVectorInCluster);	//[k] specifies for each cluster how many vectors he contains = sendCounts

double* computeClusterDiameter(double *vectors,
	int    numOfVectors, int numOfClusters, int numOfDims,
	int    *vectorIndexOfCluster);

double computeClusterQuality(double **clusters, int numOfClusters, int numOfDims, double *diameters);

#endif // !__QUALITY_H_
