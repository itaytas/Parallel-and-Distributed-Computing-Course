#ifndef __IO_H
#define __IO_H

#pragma warning( disable : 4996 )
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define PRODUCT_LEN 20

double* readVectorsFromFile(const char *fileName,           //file containing the data set
	int        *numDims,            //number of dims in vectorial space
	int        *numVectors,         //number of given vectors in file
	int        *maxNumOfClusters,   //MAX number of cluster allowed according to file
	int        *iterationLimit,     //limit of k-means iteration allowed
	double     *qualityOfClusters,  //quality of clusters to find according to file
	char      **productNames);		//[numVectors] will contain all product id's


void writeClustersToFile(char   *fileName, double **clusters, int numClusters, int numDims, double quality);

#endif //__IO_H