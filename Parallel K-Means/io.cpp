/*DIRECTIONS:
------------
The file supplied to the function must be in the next format:
-------------------------------------------------------------
(line 1):N n MAX LIMIT QM (index:
N     = number of vectors in file
n 	   = vector dimension
MAX   = maximum number of clusters to generate
LIMIT = iteration limit on k-means algo
QM    = quality measure to reach)
(line 2):vectors[0][0] vectors[0][1] ... vectors[0][n-1]
...
...
vectors[N-1][0] vectors[N-1][1] ... vectors[N-1][n-1]

*** line 1 is delimited by a space.
line 2 and onward is 1 vector per line, where all its' components are space delimited.*/

#include "io.h"

double* readVectorsFromFile(const char *fileName,           //file containing the data set
	int        *numDims,            //number of dims in vectorial space
	int        *numVectors,         //number of given vectors in file
	int        *maxNumOfClusters,   //MAX number of cluster allowed according to file
	int        *iterationLimit,     //limit of k-means iteration allowed
	double      *qualityOfClusters,  //quality of clusters to find according to file
	char      **productNames)		//[numVectors] will contain all product id's
{
	int i, j;
	double *vectors;
	char productNameBuf[PRODUCT_LEN];
	FILE *f;

	f = fopen(fileName, "r");
	assert(f != NULL);

	fscanf(f, "%d %d %d %d %lf\n", numVectors, numDims, maxNumOfClusters, iterationLimit, qualityOfClusters);

	//assiging the vectors Array (1D)
	vectors = (double*)malloc((*numVectors) * (*numDims) * sizeof(double));
	assert(vectors != NULL);

	//assigning the product names array(1D) 
	productNames = (char**)malloc((*numVectors) * sizeof(char*));
	assert(productNames != NULL);

	for (i = 0; i < *numVectors; ++i)
	{
		fscanf(f, "%s ", productNameBuf);
		productNames[i] = strdup(productNameBuf);

		for (j = 0; j < *numDims; ++j)
		{
			fscanf(f, "%lf ", &vectors[j + i* (*numDims)]);
		}
		fscanf(f, "\n");
	}

	fclose(f);
	return vectors;
}

void writeClustersToFile(char   *fileName,
	double **clusters,
	int     numClusters,
	int	 numDims,
	double   quality)
{
	int i, j;

	FILE *f = fopen(fileName, "w");
	assert(f != NULL);

	fprintf(f, "Number of clusters with the best measure: K = %d\n\n", numClusters);
	fprintf(f, "The quality with the best measure: QM = %.5f\n\n", quality);
	fprintf(f, "Centers of the clusters:\n\n");

	for (i = 0; i < numClusters; ++i)
	{
		fprintf(f, "%c%d ", 'C', i + 1);

		for (j = 0; j < numDims; ++j)
		{
			fprintf(f, "%.2f ", clusters[i][j]);
		}

		fprintf(f, "\n\n");
	}
	fclose(f);
}