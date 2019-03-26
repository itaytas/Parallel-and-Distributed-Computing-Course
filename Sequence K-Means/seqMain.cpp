#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
//#include <unistd.h>     /* getopt() */

int      _debug;
#include "k_means.h"

/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {
           int     opt;
    extern char   *optarg;
    extern int     optind;
           int     i, j;
           int     isBinaryFile, is_output_timing;

           int     numClusters, numCoords, numObjs;
           int    *membership;    /* [numObjs] */
           char   *filename;
           float **objects;       /* [numObjs][numCoords] data objects */
           float **clusters;      /* [numClusters][numCoords] cluster center */
           float   threshold;
           double  timing, io_timing, clustering_timing;
           int     loop_iterations;

    /* some default values */
    _debug           = 0;
    threshold        = 0.001;
    numClusters      = 0;
    isBinaryFile     = 0;
    is_output_timing = 0;
    filename         = NULL;

	/*
    while ( (opt=getopt(argc,argv,"p:i:n:t:abdo"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'n': numClusters = atoi(optarg);
                      break;
            case 'o': is_output_timing = 1;
                      break;
            case 'd': _debug = 1;
                      break;
            case '?': usage(argv[0], threshold);
                      break;
            default: usage(argv[0], threshold);
                      break;
        }
    }
	*/

   // if (filename == 0 || numClusters <= 1) usage(argv[0], threshold);

    //if (is_output_timing) io_timing = wtime();

    /* read data points from file ------------------------------------------*/
	/*
    objects = file_read(isBinaryFile, filename, &numObjs, &numCoords);
    if (objects == NULL) exit(1);

    if (is_output_timing) {
        timing            = wtime();
        io_timing         = timing - io_timing;
        clustering_timing = timing;
    }
	*/

    /* start the timer for the core computation -----------------------------*/
    /* membership: the cluster id for each data object */
    membership = (int*) malloc(numObjs * sizeof(int));
    assert(membership != NULL);

    clusters = seq_kmeans(objects, numCoords, numObjs, numClusters, threshold,
                          membership, &loop_iterations);

    free(objects[0]);
    free(objects);

    if (is_output_timing) {
        timing            = wtime();
        clustering_timing = timing - clustering_timing;
    }

    /* output: the coordinates of the cluster centres ----------------------*/
    file_write(filename, numClusters, numObjs, numCoords, clusters,
               membership);

    free(membership);
    free(clusters[0]);
    free(clusters);

    /*---- output performance numbers ---------------------------------------*/
    if (is_output_timing) {
        io_timing += wtime() - timing;
        printf("\nPerforming **** Regular Kmeans (sequential version) ****\n");

        printf("Input file:     %s\n", filename);
        printf("numObjs       = %d\n", numObjs);
        printf("numCoords     = %d\n", numCoords);
        printf("numClusters   = %d\n", numClusters);
        printf("threshold     = %.4f\n", threshold);

        printf("Loop iterations    = %d\n", loop_iterations);

        printf("I/O time           = %10.4f sec\n", io_timing);
        printf("Computation timing = %10.4f sec\n", clustering_timing);
    }

    return(0);
}

